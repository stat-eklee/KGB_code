# https://github.com/ThiagoCF05/webnlg/blob/master/entry.py


class Entry():
    def __init__(self, category, eid, size, originaltripleset, modifiedtripleset, entitymap, lexEntries):
        self.category = category
        self.eid = eid
        self.size = size
        self.originaltripleset = originaltripleset
        self.modifiedtripleset = modifiedtripleset
        self.lexEntries = lexEntries
        self.entitymap = entitymap

    def entitymap_to_dict(self):
        return dict(map(lambda tagentity: tagentity.to_tuple(), self.entitymap))

class Triple():
    def __init__(self, subject, predicate, object):
        self.subject = subject
        self.predicate = predicate
        self.object = object

    def __repr__(self):
        subj = self.subject.replace('"', r'\"')
        pred = self.predicate.replace('"', r'\"')
        obj = self.object.replace('"', r'\"')
        return 'Triple.parse("%s | %s | %s")' % (subj, pred, obj)

    def __str__(self):
        return '|'.join((self.subject, self.predicate, self.object))

    def __eq__(self, other):
        return self.subject == other.subject and self.predicate == other.predicate and self.object == other.object


    @staticmethod
    def parse(text):
        subj, pred, obj = text.split('|')
        return Triple(subj.strip(), pred.strip(), obj.strip())


class Lex():
    def __init__(self, comment, lid, text, template, orderedtripleset=[], references=[]):
        self.comment = comment
        self.lid = lid
        self.text = text
        self.template = template
        self.tree = ''
        self.lex_template = ''
        self.orderedtripleset = orderedtripleset
        self.references = references

        # german entry
        self.text_de = ''
        self.template_de = ''
        self.tree_de = ''
        self.orderedtripleset_de = []
        self.references_de = []

class TagEntity():
    def __init__(self, tag, entity):
        self.tag = tag
        self.entity = entity

    def to_tuple(self):
        return (self.tag, self.entity)

class Reference():
    def __init__(self, tag, entity, refex, number, reftype):
        self.tag = tag
        self.entity = entity
        self.refex = refex
        self.number = number
        self.reftype = reftype
