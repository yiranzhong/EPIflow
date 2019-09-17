import torch
from torch.autograd import Variable

def normalize2dpoints(X,Y):    
    meanX = torch.mean(X,0)
    meanY = torch.mean(Y,0)
    newpX = X-meanX
    newpY = Y-meanY
    dist  = torch.sqrt(newpX**2 + newpY**2)
    meandist = torch.mean(dist,0)
    scale = 1.4142135623730951/meandist
    
    newX = scale*(X-meanX)
    newY = scale*(Y-meanY)
    T = torch.FloatTensor([[scale, 0, -scale*meanX], [0, scale, -scale*meanY],[0,0,1]]).cuda().double()
    return newX, newY, T

def random_select_points(x,y,x_,y_,samples=10):
    idx=torch.randperm(x.shape[0])
    x=x[idx[:samples],:]
    y=y[idx[:samples],:]
    x_=x_[idx[:samples],:]
    y_=y_[idx[:samples],:]
    return x,y,x_,y_

def subspace_loss(flow,mask):
    B, _, H, W = flow.size()
    xx = Variable(torch.arange(0, W).view(1,-1).repeat(H,1).cuda())
    yy = Variable(torch.arange(0, H).view(-1,1).repeat(1,W).cuda())
    grid_x = xx.view(1,1,H,W).repeat(B,1,1,1).float()
    grid_y = yy.view(1,1,H,W).repeat(B,1,1,1).float()
    
    flow_u = flow[:,0,:,:].unsqueeze(1)
    flow_v = flow[:,1,:,:].unsqueeze(1)
    
    pos_x = grid_x + flow_u
    pos_y = grid_y + flow_v

    inside_x = (pos_x <= (W-1)) & (pos_x >= 0.0)
    inside_y = (pos_y <= (H-1)) & (pos_y >= 0.0)
    
    inside = inside_x & inside_y & mask
    
    loss = 0
    for i in range(B):
	grid_x_i = grid_x[i,:,:,:]
	grid_y_i = grid_y[i,:,:,:]
	pos_x_i = pos_x[i,:,:,:]
	pos_y_i = pos_y[i,:,:,:]
	inside_i= inside[i,:,:,:]
	
	if inside_i.sum()>2000:
	  x  = torch.masked_select(grid_x_i, inside_i).view(-1,1)
	  y  = torch.masked_select(grid_y_i, inside_i).view(-1,1)
	  x_ = torch.masked_select(pos_x_i, inside_i).view(-1,1)
	  y_ = torch.masked_select(pos_y_i, inside_i).view(-1,1)
	  x, y, x_, y_ = random_select_points(x,y,x_,y_,samples=2000)
	  o  = torch.ones_like(x)
	  x, y, x_, y_ = x/W, y/W, x_/W, y_/W
	  X  = torch.cat((x,x,x,y,y,y,o,o,o),1).permute(1,0)  
	  X_ = torch.cat((x_,y_,o,x_,y_,o,x_,y_,o),1).permute(1,0)
	  
	  M  = X * X_
	  
	  lambda1 = 10
	  MTM = lambda1 * M.permute(1,0).mm(M)
	  I = torch.eye(MTM.size()[0]).cuda()
	  temp1 = torch.inverse((I + MTM))
	  C = temp1.mm(MTM)
	  C2 = C**2
	  loss1 = torch.sum(C2.view(-1,1),dim=0)
	  temp2 = M.mm(C)-M
	  temp2 = temp2**2
	  
	  loss2 = lambda1 * torch.sum(temp2.view(-1,1),dim=0)
	  loss +=  (loss1 + loss2)
	else:
	  loss += 0.0001
        
    return loss/B

def lowrank_loss(flow,mask):
    B, _, H, W = flow.size()
    xx = Variable(torch.arange(0, W).view(1,-1).repeat(H,1).cuda())
    yy = Variable(torch.arange(0, H).view(-1,1).repeat(1,W).cuda())
    grid_x = xx.view(1,1,H,W).repeat(B,1,1,1).float()
    grid_y = yy.view(1,1,H,W).repeat(B,1,1,1).float()
    
    flow_u = flow[:,0,:,:].unsqueeze(1)
    flow_v = flow[:,1,:,:].unsqueeze(1)
    
    pos_x = grid_x + flow_u
    pos_y = grid_y + flow_v

    inside_x = (pos_x <= (W-1)) & (pos_x >= 0.0)
    inside_y = (pos_y <= (H-1)) & (pos_y >= 0.0)
    
    inside = inside_x & inside_y & mask
    
    loss = 0
    for i in range(B):
        grid_x_i = grid_x[i,:,:,:]
        grid_y_i = grid_y[i,:,:,:]
        pos_x_i = pos_x[i,:,:,:]
        pos_y_i = pos_y[i,:,:,:]
        inside_i= inside[i,:,:,:]
	if inside_i.sum()>2000:  
	  x  = torch.masked_select(grid_x_i, inside_i).view(-1,1)
	  y  = torch.masked_select(grid_y_i, inside_i).view(-1,1)

	  x_ = torch.masked_select(pos_x_i, inside_i).view(-1,1)
	  y_ = torch.masked_select(pos_y_i, inside_i).view(-1,1)
	  x, y, x_, y_ = random_select_points(x,y,x_,y_,samples=2000)
	  o  = torch.ones_like(x)
	  x, y , T = normalize2dpoints(x,y)
	  x_,y_, T_= normalize2dpoints(x_,y_)
	  
	  X  = torch.cat((x,x,x,y,y,y,o,o,o),1).permute(1,0)
      
	  X_ = torch.cat((x_,y_,o,x_,y_,o,x_,y_,o),1).permute(1,0)

	  XX = torch.cat((x,y,o),1).permute(1,0).double()
	  XX_= torch.cat((x_,y_,o),1).permute(1,0).double()

	  M  = X * X_
	  
	  U, S, V = torch.svd(M.permute(1,0)) 

	  loss += torch.sum(S.abs())/x.size()[0]
        else:
          loss += 0.0001
        
    return loss.float()