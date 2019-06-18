import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from loss import *


# -----------------------OcvConv----------------------------------
class SELayer(nn.Module):
	def __init__(self, channel, reduction = 16) :
		super(SELayer, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		channel_num = max(channel // reduction, 8)
		self.weight = nn.Sequential(
			nn.Conv2d(channel, channel_num, 1),
			nn.ReLU(inplace = True),
			nn.Conv2d(channel_num, channel, 1),
			nn.Sigmoid(),
			)

	def forward(self, x):
		y = self.avg_pool(x)
		y = self.weight(y)

		return x * y

class FirstOctConv(nn.Module):
	'''
	衔接正常卷积出来的特征图，其输出中，alpha*ch_in=8是低频部分，(1-alpha)*ch_in=56是高频部分
	'''
	def __init__(self, ch_in, ch_out, settings):
		super(FirstOctConv, self).__init__()
		alpha_in, alpha_out = settings # (0, 0.125)

		hf_ch_in = int(ch_in * (1 - alpha_in)) # 64*(1-0) = 64
		hf_ch_out = int(ch_out * (1 - alpha_out))# 64*(1-0.125) = 56

		# lf_ch_in = ch_in - hf_ch_in # 64-64 = 0
		# lf_ch_out = ch_out - hf_ch_out # 64-56 = 8
		lf_ch_in = int(ch_in * alpha_in) # 64*0 = 0
		lf_ch_out = int(ch_out * alpha_out) # 64-56 = 8

		self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
		self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

		self.conv_h2h = nn.utils.weight_norm(nn.Conv2d(hf_ch_in, hf_ch_out, kernel_size=(3, 3), padding=(1, 1), bias=False), name = 'weight', dim = 0) # 64--->56
		self.conv_h2l = nn.utils.weight_norm(nn.Conv2d(hf_ch_in, lf_ch_out, kernel_size=(3, 3), padding=(1, 1), bias=False), name = 'weight', dim = 0) # 64--->8
		# self.conv_l2h = nn.Conv2d(lf_ch_in, hf_ch_out, kernel_size=(3, 3), padding=(1, 1), bias=False)
		# self.conv_l2l = nn.Conv2d(lf_ch_in, lf_ch_out, kernel_size=(3, 3), padding=(1, 1), bias=False)


	def forward(self, x):
		hf_data = x
		h2h_conv = self.conv_h2h(hf_data)
		h2l_pool = self.downsample(hf_data)
		h2l_pool_conv = self.conv_h2l(h2l_pool)

		out_h = h2h_conv
		out_l = h2l_pool_conv
		return out_h, out_l

class OctConv(nn.Module):
	'''
	对FirstOctConv产生的两个输出：高频+低频进行卷积和融合，并产生两个输出
	'''
	def __init__(self, ch_in, ch_out, settings):
		super(OctConv, self).__init__()

		alpha_in, alpha_out = settings # (0.125, 0.125)

		hf_ch_in = int(ch_in * (1 - alpha_in)) # 64*(1-0.125) = 56
		hf_ch_out = int(ch_out * (1 - alpha_out))# 64*(1-0.125) = 56

		lf_ch_in = int(ch_in * alpha_in) # 64*0.128 = 8
		lf_ch_out = int(ch_out * alpha_out) # 64*0.125 = 8

		self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
		self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

		self.conv_h2h = nn.utils.weight_norm(nn.Conv2d(hf_ch_in, hf_ch_out, kernel_size=(3, 3), padding=(1, 1), bias=False), name = 'weight', dim = 0)  # 56--->56
		self.conv_h2l = nn.utils.weight_norm(nn.Conv2d(hf_ch_in, lf_ch_out, kernel_size=(3, 3), padding=(1, 1), bias=False), name = 'weight', dim = 0)  # 56--->8
		self.conv_l2h = nn.utils.weight_norm(nn.Conv2d(lf_ch_in, hf_ch_out, kernel_size=(3, 3), padding=(1, 1), bias=False), name = 'weight', dim = 0)  # 8--->56
		self.conv_l2l = nn.utils.weight_norm(nn.Conv2d(lf_ch_in, lf_ch_out, kernel_size=(3, 3), padding=(1, 1), bias=False), name = 'weight', dim = 0)  # 8--->8

	# def forward(self, hf_data, lf_data): # 输入 cb cr
	def forward(self, x):  # 输入 cb cr
		# if type(x) is tuple: #如何判断
		hf_data, lf_data = x
		h2h_conv = self.conv_h2h(hf_data)
		h2l_pool = self.downsample(hf_data)
		h2l_pool_conv = self.conv_h2l(h2l_pool)

		l2h_conv = self.conv_l2h(lf_data)
		l2h_upsample_conv = self.upsample(l2h_conv)
		l2l_conv = self.conv_l2l(lf_data)

		out_h = h2h_conv + l2h_upsample_conv  # 替换concat+conv
		out_l = l2l_conv + h2l_pool_conv      # 替换concat+conv
		return out_h, out_l

class LastOctConv(nn.Module):
	'''
	对OctConv产生的高频和低频，高频卷积+低频上采样，融合之后，只产生一个输出
	'''
	def __init__(self, ch_in, ch_out, settings):
		super(LastOctConv, self).__init__()

		alpha_in, alpha_out = settings # (0.125, 0)

		hf_ch_in = int(ch_in * (1 - alpha_in)) # 64*(1-0.125) = 56
		hf_ch_out = int(ch_out * (1 - alpha_out))# 64*(1-0) = 64

		lf_ch_in = int(ch_in * alpha_in) # 64*0.128 = 8
		lf_ch_out = int(ch_out * alpha_out) # 64*0 = 0

		self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
		self.upsample = nn.Upsample(scale_factor=2, mode='nearest') # bilinear

		self.conv_h2h = nn.utils.weight_norm(nn.Conv2d(hf_ch_in, hf_ch_out, kernel_size=(3, 3), padding=(1, 1), bias=False), name = 'weight', dim = 0)  # 56--->64
		# self.conv_h2l = nn.Conv2d(hf_ch_in, lf_ch_out, kernel_size=(3, 3), padding=(1, 1), bias=False)  # 56--->8
		self.conv_l2h = nn.utils.weight_norm(nn.Conv2d(lf_ch_in, hf_ch_out, kernel_size=(3, 3), padding=(1, 1), bias=False), name = 'weight', dim = 0)  # 8--->64
		# self.conv_l2l = nn.Conv2d(lf_ch_in, lf_ch_out, kernel_size=(3, 3), padding=(1, 1), bias=False)  # 8--->8

	def forward(self, x): # 输入 cb cr
		hf_data, lf_data = x
		h2h_conv = self.conv_h2h(hf_data)

		l2h_conv = self.conv_l2h(lf_data)
		l2h_upsample_conv = self.upsample(l2h_conv)
		# l2l_conv = self.conv_l2l(lf_data)

		out_h = h2h_conv + l2h_upsample_conv  # 替换concat+conv
		# out_l = l2l_conv + h2l_pool_conv      # 替换concat+conv
		return out_h

class OctConv_WN_AC(nn.Module):
	'''
	进行WN和激活函数操作，WN未加上（参数不知道怎么传）
	'''
	def __init__(self, ch_in, ch_out, settings, act=True):
		super(OctConv_WN_AC, self).__init__()

		self.flage = act
		self.conv = OctConv(ch_in, ch_out, settings)

		self.act = nn.ReLU()

	def forward(self, x):
		hf_data, lf_data = self.conv(x)
		if self.flage:
			hf_data = self.act(hf_data)
			lf_data = self.act(lf_data)
		else:
			hf_data = hf_data
			lf_data = lf_data
		return hf_data, lf_data
