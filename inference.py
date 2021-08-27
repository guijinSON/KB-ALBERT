from utils import Zero_Shot_TC

def binary_score(score,label1,label2):
    text=''
    if score > 50:
        text = f'강한 {label1}'
    elif score>0:
        text = f'약한 {label1}'
    elif -50<score<0:
        text = f'약한 {label2}'
    elif score<-50:
        text = f'강한 {label2}'
    return text

def Risk_Tolerance(text,return_logits=False):
    result = Zero_Shot_TC(text,['위험 회피형','위험 추구형'],"이 문장은 {label} 성격을 가진다.")
    risk_score = result['위험 추구형'] - result['위험 회피형']
    if return_logits:
        return binary_score(risk_score,'위험 추구형','위험 회피형'),risk_score
    return binary_score(risk_score,'위험 추구형','위험 회피형')
