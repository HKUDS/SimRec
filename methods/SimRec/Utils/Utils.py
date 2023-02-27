import torch as t
import torch.nn.functional as F

def innerProduct(usrEmbeds, itmEmbeds):
	return t.sum(usrEmbeds * itmEmbeds, dim=-1)

def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)
	
def calcRegLoss(params=None, model=None):
	ret = 0
	if params is not None:
		for W in params:
			ret += W.norm(2).square()
	if model is not None:
		for W in model.parameters():
			ret += W.norm(2).square()
	# ret += (model.usrStruct + model.itmStruct)
	return ret

def infoNCE(embeds1, embeds2, nodes, temp):
	embeds1 = F.normalize(embeds1 + 1e-8, p=2)
	embeds2 = F.normalize(embeds2 + 1e-8, p=2)
	pckEmbeds1 = embeds1[nodes]
	pckEmbeds2 = embeds2[nodes]
	nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
	deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
	return (-t.log(nume / deno)).mean()

def KLDiverge(tpreds, spreds, distillTemp):
	tpreds = (tpreds / distillTemp).sigmoid()
	spreds = (spreds / distillTemp).sigmoid()
	return -(tpreds * (spreds + 1e-8).log() + (1 - tpreds) * (1 - spreds + 1e-8).log()).mean()

def pointKLDiverge(tpreds, spreds):
	return -(tpreds * spreds.log()).mean()