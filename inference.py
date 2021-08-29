from utils import Zero_Shot_TC,binary_score
import torch

def Financial_Understanding(text,labels,tokenizer,model,weight_path,return_logits=False):
    weight = torch.load(weight_path,map_location=torch.device('cpu')) #2_0
    model.load_state_dict(weight)
    result = {}
    hypothesis = [f'해당 문장은 {label}에 대해 설명하고 있다.' for label in labels]
    for label,h in zip(labels,hypothesis):
        x = tokenizer.encode(text, h, max_length=64, return_tensors='pt',truncation=True)
        logits = model(x).logits
        probs = logits.softmax(dim=1)
        entailment = probs.detach()[:,0]
        contradiction = probs.detach()[:,2]
        result[label] = entailment
    output = dict(zip(result.values(),result.keys()))
    m = max(output.keys())
    if return_logits:
        return output[m] , contradiction
    return f'해당 문장은 {output[m]}에 대한 이해를 보여주고 있다.'

def Risk_Tolerance(text,return_logits=False):
    result = Zero_Shot_TC(text,['위험 회피형','위험 추구형'],"이 문장은 {label} 성격을 가진다.")
    risk_score = result['위험 추구형'] - result['위험 회피형']
    if return_logits:
        return binary_score(risk_score,'위험 추구형','위험 회피형'),risk_score
    return binary_score(risk_score,'위험 추구형','위험 회피형')


