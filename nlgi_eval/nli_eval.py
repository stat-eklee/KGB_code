#!/usr/bin/env python3

import json
import torch
import re
import numpy as np
import pandas as pd
import sklearn
import scipy

from argparse import ArgumentParser
#from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from logzero import logger

from external.tgen_data import DA
from external.e2e_metrics_tsv import read_tsv
from external.webnlg_entry import Triple
import external.webnlg_parser as webnlg_parser


TEMPLATE_PATHS = {
    'e2e': 'templates/e2e_templates.json',
    'webnlg': 'templates/webnlg_templates.json',
}

# the templates are slightly different from what E2E evaluation produced => remap
TEMPLATE_REMAP = {
    'eat_type': 'eatType',
    'rating': 'customer rating',
    'family_friendly': 'familyFriendly',
    'price_range': 'priceRange',
}


class TripleJSONEnc(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, Triple):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


class Evaluator:

    def __init__(self, file_format, use_templates=True, e2e_ignore_restaurant=False, model="bert-base-uncased",tokenizer="bert-base-uncased"):

        self.use_gpu = torch.cuda.is_available()
        logger.debug('Use GPU: %r' % self.use_gpu)

        # load roberta
        logger.debug('Loading models...')
        self.tokenizer = tokenizer 
        self.model = model #AutoModelForSequenceClassification.from_pretrained('/home/ekl/Desktop/eklee/models/roberta-large-mnli')
        if self.use_gpu:
            self.model.to('cuda')

        # load templates
        if use_templates:
            logger.debug('Loading templates...')
            with open(TEMPLATE_PATHS[file_format], 'r', encoding='UTF-8') as fh:
                self.templates = json.load(fh)
        else:
            self.templates = {}
        # set parse method
        if file_format == 'webnlg':
            self.parse_data = self.parse_webnlg
            self.check_with_gold = self.check_with_gold_webnlg
            self.e2e_ignore_restaurant = False  # E2E-only setting
        elif file_format == 'e2e':
            self.parse_data = self.parse_e2e
            self.check_with_gold = self.check_with_gold_e2e
            self.e2e_ignore_restaurant = e2e_ignore_restaurant
        logger.debug('Ready.')

    def triples_to_templates(self, tripleset):
        """Convert given triples to templates, return a list of flags telling which of them
        should be optional (only used to ignore eatType=restaurant with the given global setting)."""
        output = []
        optionals = []
        for triple in tripleset:
            if triple.predicate not in self.templates:
                # if template isn't found, check remapping first
                if triple.predicate in TEMPLATE_REMAP and TEMPLATE_REMAP[triple.predicate] in self.templates:
                    template = self.templates[TEMPLATE_REMAP[triple.predicate]]
                else:  # if remapping doesn't work either, create a backoff template
                    template = 'The %s of <subject> is <object>.' % triple.predicate
                    template = re.sub('([a-z])([A-Z])', r"\1 \2", template)  # get rid of camel case
                    logger.warn('Created backoff template for %s' % (triple.predicate))
                    self.templates[triple.predicate] = template
            else:  # take the template
                template = self.templates[triple.predicate]
            if isinstance(template, dict):
                template = template[triple.object]
            if isinstance(template, list):
                template = template[0]  # XXX don't take the first, but the best template
            template = template.replace('<subject>', triple.subject)
            obj_str = re.sub('^["\'](.*)["\']$', r'\1', triple.object)  # remove quotes around values
            template = template.replace('<object>', obj_str)
            template = template.replace('_', ' ')
            output.append(template)
            if self.e2e_ignore_restaurant and triple.predicate in ['eat_type', 'eatType'] and obj_str == 'restaurant':
                optionals.append(True)  # if set so, eatType=restaurant is optional
            else:
                optionals.append(False)  # default: not optional
        return output, optionals

    def roberta_classify(self, a, b):
        inputs = self.tokenizer("%s </s></s> %s" % (a, b), return_tensors="pt")
        labels = torch.tensor([1]).unsqueeze(0)  # batch size = 1
        if self.use_gpu:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
        # _, output = self.model(**inputs, labels=labels)  # ignoring loss
        output = self.model(**inputs, labels=labels).logits  # ignoring loss
        #output = self.model(**inputs, labels=labels)[1]

        if self.use_gpu:
            output = output.cpu()
        output = torch.nn.Softmax(dim=1)(output).detach().numpy()[0]
        print(output)
        return output

    def check_inst(self, mr, sent):
        # return templates + potentially mark some as optional (eatType=restaurant)
        templs, opts = self.triples_to_templates(mr)

        logger.debug("%s\nTEMP: %s\nSENT: %s" % ("  ++  ".join([str(t) for t in mr]), ' '.join(templs), sent))
        ent_confs = []
        # mr -> sent
        mr2sent = self.roberta_classify(' '.join(templs), sent)
        #raw_results = {'mr2sent': "C: %.4f N: %.4f E: %.4f" % tuple(mr2sent)}
        #logger.debug("--> " + raw_results['mr2sent'])
        raw_results = {'mr2sent': "E: %.4f N: %.4f C: %.4f" % tuple(mr2sent)}

        ent_confs.append(float(mr2sent[2]))
        #mr2sent = ['C', 'N', 'E'][np.argmax(mr2sent)]
        mr2sent = ['E', 'N', 'C'][np.argmax(mr2sent)]
        # sent -> mr (remove optional templates/triples, where optional is True)
        sent2mr = self.roberta_classify(sent, ' '.join([t for t, o in zip(templs, opts) if not o]))
        raw_results['sent2mr'] = "C: %.4f N: %.4f E: %.4f" % tuple(sent2mr)
        logger.debug("<-- " + raw_results['sent2mr'])
        ent_confs.append(float(sent2mr[2]))
        #sent2mr = ['C', 'N', 'E'][np.argmax(sent2mr)]
        sent2mr = ['E', 'N', 'C'][np.argmax(sent2mr)]

        if sent2mr == 'E' and mr2sent == 'E':
            output = 'OK'
        elif sent2mr == 'E':
            output = 'hallucination'
        elif mr2sent == 'E':
            output = 'omission'
        else:
            output = 'hallucination+omission'
        logger.debug(output)

        # sent -> mr for individual slots
        omitted = []
        for triple, templ, optional in zip(mr, templs, opts):
            if optional:  # skip optional templates/triples
                continue
            sent2triple = self.roberta_classify(sent, templ)
            # ent_confs.append(float(sent2triple[2]))
            ent_confs.append(float(sent2triple[0]))

            #sent2triple = ['C', 'N', 'E'][np.argmax(sent2triple)]
            sent2triple = ['E', 'N', 'C'][np.argmax(sent2triple)]

            if sent2triple != 'E':
                omitted.append(triple)

        if omitted:
            #logger.debug('Omitted: %s' % '  ++  '.join([str(t) for t in omitted]))
            # override the global decision -- this is more fine-grained
            output = 'hallucination+omission' if 'hallucination' in output else 'omission'

        return output, min(ent_confs), omitted, raw_results, templs

    def console(self):
        mr_text = input('Input MR (triples divided by |, separated by ++):\n  > ')
        if not mr_text:
            return False
        mr = [Triple.parse(s.strip()) for s in mr_text.split('++')]
        sent = input('Input text:\n  > ')
        if not sent:
            return False
        self.check_inst(mr, sent)
        return True

    def parse_e2e(self, fname):
        mrs, sents = read_tsv(fname)
        mrs = [self.da_to_triples(DA.parse_diligent_da(mr)) for mr in mrs]
        return [(mr, sent) for mr, sent in zip(mrs, sents)]

    def da_to_triples(self, da):
        # convert a TGen DA into WebNLG style triples
        name = [dai for dai in da if dai.slot == 'name'][0].value
        return [Triple(name, dai.slot, dai.value) for dai in da if dai.slot != 'name']

    def parse_webnlg(self, fnames):
        if ',' in fnames:
            # split input & output filenames
            input_fname, output_fname = fnames.split(',')
            # parse input XML triples
            mrs = list(webnlg_parser.parse(input_fname))
            mrs = [mr.modifiedtripleset for mr in mrs]
            # read output sentences
            with open(output_fname, 'r', encoding='UTF-8') as fh:
                sents = [line.strip() for line in fh.readlines()]
        else:
            data = pd.read_csv(fnames, sep=',', encoding='UTF-8').to_dict('records')
            mrs = [[Triple.parse(t) for t in inst['mr'].split('<br>')] for inst in data]
            sents = [inst['text'] for inst in data]
        return [(mr, sent) for mr, sent in zip(mrs, sents)]

    def eval_file(self, in_fname):
        """Main method. Evaluate a given file."""
        data = self.parse_data(in_fname)
        outputs = []
        for mr, sent in data:
            result, OK_conf, omitted, raw_results, templs = self.check_inst(mr, sent)
            outputs.append({'mr': mr,
                            'templates': ' '.join(templs),
                            'sent': sent,
                            'result': result,
                            'OK_confidence': OK_conf,
                            'omitted': omitted,
                            'raw_results': raw_results})
        return outputs

    def load_preds_from_file(self, in_fname, in_fname_preds):
        """Instead of doing predictions, load them from a file -- to evaluate an external classifier
        or rerun evaluation without re-classifying."""
        data = self.parse_data(in_fname)
        # load a JSON prediction file, expecting the proper format
        if in_fname_preds.endswith('.json'):
            with open(in_fname_preds, 'r', encoding='UTF-8') as fh:
                preds = json.load(fh)['predictions']
            assert len(preds) == len(data)
            for pred in preds:
                if 'error' in pred:
                    del pred['error']
            return preds
        # parse a TSV file: result/omitted/raw results assumed as columns
        else:
            with open(in_fname_preds, 'r', encoding='UTF-8') as fh:
                preds = [l.strip().split("\t") for l in fh.readlines()]
            assert len(preds) >= len(data) and len(preds) <= len(data) + 1
            if len(preds) > len(data):  # skip header
                preds = preds[1:]
            outputs = []
            for (mr, sent), (result, omitted, raw_results) in zip(data, preds):
                outputs.append({'mr': mr,
                                'sent': sent,
                                'result': result,
                                'omitted': omitted,
                                'raw_results': raw_results})
        return outputs

    def compute_metrics(self, y_pred, y_gold, select=None, fine=False):
        """Compute accuracy and confusion matrix. Compute recall for "not OK" if not in fine mode."""
        labels_order = ['OK', 'not OK']
        row_labels=['g_OK', 'g_not']
        col_labels=['p_OK', 'p_not']
        if fine:
            labels_order = ['OK', 'hallucination', 'omission', 'hallucination+omission']
            row_labels = ['g_OK', 'g_hal', 'g_om', 'g_h+o']
            col_labels = ['p_OK', 'p_hal', 'p_om', 'p_h+o']

        if select:
            y_gold = [y for (y, sel) in zip(y_gold, select) if sel]
            y_pred = [y for (y, sel) in zip(y_pred, select) if sel]

        acc = sklearn.metrics.accuracy_score(y_gold, y_pred)
        conf_matrix = sklearn.metrics.confusion_matrix(y_gold, y_pred, labels=labels_order)
        conf_matrix = pd.DataFrame(conf_matrix, index=row_labels, columns=col_labels)

        logger.info('Accuracy: %.4f' % acc)
        logger.info('Confusion matrix:\n%s' % str(conf_matrix))
        results = {'accuracy': acc,
                   'conf_matrix': conf_matrix.to_dict('index'),
                   'N': len(y_pred)}
        if not fine:
            rec = sklearn.metrics.recall_score(y_gold, y_pred, pos_label='not OK')
            pre = sklearn.metrics.precision_score(y_gold, y_pred, pos_label='not OK')
            f1 = sklearn.metrics.f1_score(y_gold, y_pred, pos_label='not OK')
            logger.info('Recall: %.4f' % rec)
            logger.info('Precision: %.4f' % pre)
            logger.info('F1: %.4f' % f1)
            results['recall'] = rec
            results['precision'] = pre
            results['f1'] = f1

        return results

    def check_with_gold_webnlg(self, preds, gold_fname):
        """Evaluation for WebNLG (against the "semantics" column in human eval)."""
        golds = pd.read_csv(gold_fname, sep=',', encoding='UTF-8').to_dict('records')

        max_len = 0
        for idx, (pred, gold) in enumerate(zip(preds, golds)):
            max_len = max(max_len, len(pred['mr']))
            # adding debug info
            if 'team' in gold:
                pred['sys_id'] = gold['team']
            pred['gold_human_rating'] = gold['semantics']
            if (pred['result'] == 'OK') != (gold['semantics'] >= 2.5):
                pred['error'] = True

        results = {'predictions': preds}
        # metrics
        for threshold in [2.5, 2.0]:
            y_pred = ['OK' if inst['result'] == 'OK' else 'not OK' for inst in preds]
            y_gold = ['OK' if inst['semantics'] >= threshold else 'not OK' for inst in golds]
            logger.info('Threshold %.1f...' % threshold)
            results['metrics @ %.1f' % threshold] = self.compute_metrics(y_pred, y_gold)

            for mr_len in range(1, max_len + 1):
                select = [len(inst['mr']) == mr_len for inst in preds]
                if any(select):
                    logger.info('MR length == %d...' % mr_len)
                    results['metrics @ %.1f' % threshold]['mr_len==%d' % mr_len] = self.compute_metrics(y_pred, y_gold, select=select)

        # correlation with humans
        conf_rho, conf_rho_p = scipy.stats.spearmanr([inst['OK_confidence'] for inst in preds],
                                                     [inst['semantics'] for inst in golds])
        logger.info('Spearman correlation of OK_confidence with human ratings: %.4f (p=%.4f)'
                    % (conf_rho, conf_rho_p))
        results['OK_correlation'] = {'rho': conf_rho, 'p_value': conf_rho_p}
        return results

    def check_with_gold_e2e(self, preds, gold_fname):
        """Evaluation for E2E (against the slot error automatic script predictions)."""
        golds = pd.read_csv(gold_fname, sep='\t', encoding='UTF-8').to_dict('records')
        # adapt format to our output
        for gold in golds:
            gold['diff'] = json.loads(gold['diff'])
            if self.e2e_ignore_restaurant:
                diff_eat_type = gold['diff'].get('eat_type', gold['diff'].get('eatType', {}))
                if diff_eat_type.get('restaurant', 0) > 0:
                    gold['added'] -= 1
                elif diff_eat_type.get('restaurant', 0) < 0:
                    gold['missing'] -= 1
            result = 'OK'
            if gold['valerr']:
                result = 'hallucination+omission'
            else:
                if gold['added']:
                    result = 'hallucination'
                if gold['missing']:
                    result = 'hallucination+omission' if gold['added'] else 'omission'
            gold['result'] = result

        max_len = 0
        for idx, (pred, gold) in enumerate(zip(preds, golds)):
            max_len = max(max_len, len(pred['mr']))
            # adding debug info
            if 'sys_id' in gold:
                pred['sys_id'] = gold['sys_id']
            pred['gold_result'] = gold['result']
            if pred['result'] != gold['result']:
                pred['error'] = True
            pred['gold_diff'] = gold['diff']

        results = {'predictions': preds}
        # metrics + rough metrics

        y_pred = [inst['result'] for inst in preds]
        y_gold = [inst['result'] for inst in golds]
        logger.info('Computing Fine metrics...')
        results['metrics_fine'] = self.compute_metrics(y_pred, y_gold, fine=True)
        for mr_len in range(1, max_len + 1):
            select = [len(inst['mr']) == mr_len for inst in preds]
            if any(select):
                logger.info('MR length == %d...' % mr_len)
                results['metrics_fine']['mr_len==%d' % mr_len] = self.compute_metrics(y_pred, y_gold, select=select, fine=True)

        y_pred = ['OK' if inst == 'OK' else 'not OK' for inst in y_pred]
        y_gold = ['OK' if inst == 'OK' else 'not OK' for inst in y_gold]

        logger.info('Computing Rough metrics...')
        results['metrics_rough'] = self.compute_metrics(y_pred, y_gold)
        for mr_len in range(1, max_len + 1):
            select = [len(inst['mr']) == mr_len for inst in preds]
            if any(select):
                logger.info('MR length == %d...' % mr_len)
                results['metrics_rough']['mr_len==%d' % mr_len] = self.compute_metrics(y_pred, y_gold, select=select)

        return results


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--model_name_or_path', default="/home/ekl/Desktop/eklee/models/roberta-large-mnli")
    ap.add_argument('--type', '-t', choices=['webnlg', 'e2e'], help='File format/domain templates setting', required=True)
    ap.add_argument('--eval', '-e', action='store_true', help='Input file has gold-standard predictions for evaluation')
    ap.add_argument('--console', '-c', action='store_true', help='Enter console mode (input & output files not required/ignored)')
    ap.add_argument('--pred-from-file', '-p', type=str, help='Do not predict, load predictions from the given file (i.e. evaluate only)')
    ap.add_argument('--no-templates', dest='templates', action='store_false', help='Do not use any preloaded templates, use backoff only')
    ap.add_argument('--e2e-ignore-restaurant', action='store_true', help='Be lenient about the eatType=restaurant value')

    args, _ = ap.parse_known_args()

    if not args.console:  # only require file arguments if not working on a console
        ap.add_argument('input_file', type=str, help='Input file(s) to check')
        ap.add_argument('output_file', type=str, help='Output file')

    args = ap.parse_args()

    from adapters import AutoAdapterModel
    # testing 
    # model = AutoAdapterModel.from_pretrained("gpt2")
    # adapter_name = model.load_adapter("AdapterHub/gpt2_nli_multinli_houlsby")
    # model.set_active_adapters(adapter_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    # ['E', 'N', 'C'] label 순서 주의
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=3)

    logger.debug('Starting...')
    # base 
    evaluator = Evaluator(args.type, args.templates, args.e2e_ignore_restaurant, model=model, tokenizer=tokenizer)
    # backoff
    # evaluator = Evaluator(args.type, args.templates, args.e2e_ignore_restaurant, model=model, tokenizer=tokenizer)

    if args.console:
        while evaluator.console():
            pass
    else:
        if args.pred_from_file:
            predictions = evaluator.load_preds_from_file(args.input_file, args.pred_from_file)
        else:
            predictions = evaluator.eval_file(args.input_file)

        logger.debug('Evaluation...')
        #if args.eval:
        predictions = evaluator.check_with_gold(predictions, args.input_file)

        logger.debug('Writing output...')
        with open(args.output_file, 'w', encoding='UTF-8') as fh:
            json.dump(predictions, fh, ensure_ascii=False, indent=4, cls=TripleJSONEnc)
        logger.debug('Done.')