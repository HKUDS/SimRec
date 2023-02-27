from statistics import mean
from sys import stdout
from time import time
from tkinter import mainloop
import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import infoNCE, KLDiverge, pairPredict, calcRegLoss
import datetime

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

class Model(nn.Module):
	def __init__(self, teacher):
		super(Model, self).__init__()

		self.teacher = teacher
		self.student = MLPNet()
	
	def forward(self):
		pass

	def calcLoss(self, adj, ancs, poss, negs, opt):
		uniqAncs = t.unique(ancs)
		uniqPoss = t.unique(poss)
		suEmbeds, siEmbeds = self.student()
		tuEmbeds, tiEmbeds = self.teacher(adj)
		tuEmbeds = tuEmbeds.detach()
		tiEmbeds = tiEmbeds.detach()

		rdmUsrs = t.randint(args.user, [args.topRange])#ancs
		rdmItms1 = t.randint_like(rdmUsrs, args.item)
		rdmItms2 = t.randint_like(rdmUsrs, args.item)

		# contrastive regularization for node embeds
		tEmbedsLst = self.teacher(adj, getMultOrder=True)
		highEmbeds = sum(tEmbedsLst[2:])
		highuEmbeds = highEmbeds[:args.user].detach()
		highiEmbeds = highEmbeds[args.user:].detach()
		contrastDistill = (infoNCE(highuEmbeds, suEmbeds, uniqAncs, args.tempcd) + infoNCE(highiEmbeds, siEmbeds, uniqPoss, args.tempcd)) * args.cdreg


		# prediction-level distillation
		tpairPreds = self.teacher.pairPredictwEmbeds(tuEmbeds, tiEmbeds, rdmUsrs, rdmItms1, rdmItms2)
		spairPreds = self.student.pairPredictwEmbeds(suEmbeds, siEmbeds, rdmUsrs, rdmItms1, rdmItms2)
		softTargetDistill = KLDiverge(tpairPreds, spairPreds, args.tempsoft) * args.softreg


		preds = self.student.pointPosPredictwEmbeds(suEmbeds, siEmbeds, ancs, poss)
		mainLoss = - preds.mean()

		opt.zero_grad()
		(contrastDistill).backward(retain_graph=True)
		pckUGrad1 = suEmbeds.grad[uniqAncs].detach().clone()
		pckIGrad1 = siEmbeds.grad[uniqPoss].detach().clone()
		opt.zero_grad()
		(softTargetDistill).backward(retain_graph=True)
		pckUGrad2 = suEmbeds.grad[uniqAncs].detach().clone()
		pckIGrad2 = siEmbeds.grad[uniqPoss].detach().clone()
		opt.zero_grad()
		mainLoss.backward(retain_graph=True)
		pckUGrad4 = suEmbeds.grad[uniqAncs].detach().clone()
		pckIGrad4 = siEmbeds.grad[uniqPoss].detach().clone()
		def calcGradSim(grad1, grad2, grad4):
			grad3 = grad1 + grad2
			grad1, grad2, grad3, grad4 = list(map(lambda x: F.normalize(x), [grad1, grad2, grad3, grad4]))
			sim12 = (grad1 * grad2).sum(-1)
			sim34 = (grad3 * grad4).sum(-1)
			return (sim34 <= sim12) * 1.2 + (sim34 > sim12) * 0.8
		uTaskGradSim = calcGradSim(pckUGrad1, pckUGrad2, pckUGrad4)
		iTaskGradSim = calcGradSim(pckIGrad1, pckIGrad2, pckIGrad4)

		# contrastive regularization
		selfContrast = 0
		selfContrast += (t.log(self.student.pointNegPredictwEmbeds(suEmbeds, siEmbeds, uniqAncs, args.tempsc) + 1e-5) * uTaskGradSim).mean()
		selfContrast += (t.log(self.student.pointNegPredictwEmbeds(suEmbeds, suEmbeds, uniqAncs, args.tempsc) + 1e-5) * uTaskGradSim).mean()
		selfContrast += (t.log(self.student.pointNegPredictwEmbeds(siEmbeds, siEmbeds, uniqPoss, args.tempsc) + 1e-5) * iTaskGradSim).mean()
		selfContrast *= args.screg

		# weight-decay reg
		regParams = [self.student.uEmbeds, self.student.iEmbeds]
		regLoss = calcRegLoss(params=regParams) * args.reg

		loss = mainLoss + contrastDistill + softTargetDistill + regLoss + selfContrast
		losses = {'mainLoss': mainLoss, 'contrastDistill': contrastDistill, 'softTargetDistill': softTargetDistill, 'regLoss': regLoss}
		return loss, losses

class BLMLP(nn.Module):
	def __init__(self):
		super(BLMLP, self).__init__()
		self.W = nn.Parameter(init(t.empty(args.latdim, args.latdim)))
		self.act = nn.LeakyReLU(negative_slope=0.5)
	
	def forward(self, embeds):
		pass

	def featureExtract(self, embeds):
		return self.act(embeds @ self.W) + embeds

	def pairPred(self, embeds1, embeds2):
		return (self.featureExtract(embeds1) * self.featureExtract(embeds2)).sum(dim=-1)
	
	def crossPred(self, embeds1, embeds2):
		return self.featureExtract(embeds1) @ self.featureExtract(embeds2).T

class MLPNet(nn.Module):
	def __init__(self):
		super(MLPNet, self).__init__()

		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		self.MLP = BLMLP()
		self.overallTime = datetime.timedelta(0)
	
	def forward(self):
		return self.uEmbeds, self.iEmbeds

	def pointPosPredictwEmbeds(self, uEmbeds, iEmbeds, ancs, poss):
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		nume = self.MLP.pairPred(ancEmbeds, posEmbeds)
		return nume

	def pointNegPredictwEmbeds(self, embeds1, embeds2, nodes1, temp=1.0):
		pckEmbeds1 = embeds1[nodes1]
		preds = self.MLP.crossPred(pckEmbeds1, embeds2)
		return t.exp(preds / temp).sum(-1)
	
	def pairPredictwEmbeds(self, uEmbeds, iEmbeds, ancs, poss, negs):
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		posPreds = self.MLP.pairPred(ancEmbeds, posEmbeds)
		negPreds = self.MLP.pairPred(ancEmbeds, negEmbeds)
		return posPreds - negPreds
	
	def predAll(self, pckUEmbeds, iEmbeds):
		return self.MLP.crossPred(pckUEmbeds, iEmbeds)
	
	def testPred(self, usr, trnMask):
		uEmbeds, iEmbeds = self.forward()
		allPreds = self.predAll(uEmbeds[usr], iEmbeds) * (1 - trnMask) - trnMask * 1e8
		return allPreds

class LightGCN(nn.Module):
	def __init__(self):
		super(LightGCN, self).__init__()

		self.uEmbeds = nn.Parameter(init(t.empty(args.user, args.latdim)))
		self.iEmbeds = nn.Parameter(init(t.empty(args.item, args.latdim)))
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

	def forward(self, adj, getMultOrder=False):
		embeds = t.concat([self.uEmbeds, self.iEmbeds], axis=0)
		embedsLst = [embeds]
		for gcn in self.gcnLayers:
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		embeds = sum(embedsLst)
		if not getMultOrder:
			return embeds[:args.user], embeds[args.user:]
		else:
			return embedsLst
	
	def pairPredictwEmbeds(self, uEmbeds, iEmbeds, ancs, poss, negs):
		ancEmbeds = uEmbeds[ancs]
		posEmbeds = iEmbeds[poss]
		negEmbeds = iEmbeds[negs]
		return pairPredict(ancEmbeds, posEmbeds, negEmbeds)
	
	def predAll(self, pckUEmbeds, iEmbeds):
		return pckUEmbeds @ iEmbeds.T
	
	def testPred(self, usr, trnMask, adj):
		uEmbeds, iEmbeds = self.forward(adj)
		allPreds = self.predAll(uEmbeds[usr], iEmbeds) * (1 - trnMask) - trnMask * 1e8
		return allPreds

class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	def forward(self, adj, embeds):
		return t.spmm(adj, embeds)
