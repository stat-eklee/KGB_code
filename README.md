# KGB_code
KGB Paper Training, Evaluation code

=========
bash run.sh {평가모델명} 실행 
webnlg, e2e 순서대로 origin, backoff 수행. 

# Classification
-> MNLI dataset을 Llamaclassification model로 학습시키는 code
-> 기본적으로 single gpu로 code를 짬 (multi gpu도 가능)
-> accelerate를 library를 활용.
-> script의 train.sh / inference.sh 참조.
--> train.sh에서 --model만 llama3로 대체하면 됩니다. 
--> 결과물 확인을 원한다면 inference.sh해서 accuracy 측정 가능.
※ MNLI dataset의 기본 label과 달리, 이 학습 코드는 0 : contradiction, 1: neutral, 2: entailment 임
