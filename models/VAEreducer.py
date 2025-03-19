import torch
import torch.nn as nn
import torch.nn.functional as F
class VAE(nn.Module):
    def __init__(self,latent_dim: int,):
        super(VAE, self).__init__()


        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.atten = nn.MultiheadAttention(128,4)

        # 潜在空间的线性变换
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)  # 假设图像尺寸缩小到 4x4
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # 解码器
        self.decoder_fc = nn.Linear(latent_dim, 128 * 4 * 4)
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

    def encode(self, x):
        x1 = self.encoder(x)
        x = torch.flatten(x1, start_dim=2)
        x=x.permute(0,2,1)
        x2=self.atten(x,x,x)
        x=torch.flatten(x2[0],start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar,x1

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 128, 4, 4)  # 重新形状以匹配解码器的输入
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar,x1 = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar,x1
    def loss_function(self,recons,input,mu,log_var):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        kld_weight = 1
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples
        


class UnetVAE(nn.Module):
    def __init__(self,latent_dim: int,):
        super(UnetVAE, self).__init__()


        # 编码器
        self.convblk1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),)
        self.convblk2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),)
        self.convblk3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),)
        self.convblk4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.atten = nn.MultiheadAttention(128,4)

        # 潜在空间的线性变换
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)  # 假设图像尺寸缩小到 4x4
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # 解码器
        self.decoder_fc = nn.Linear(latent_dim, 128 * 4 * 4)
        
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),)
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),)
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),)
        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x1=self.convblk1(x)
        x2=self.convblk2(x1)
        x3=self.convblk3(x2)
        x4=self.convblk4(x3)

        x = torch.flatten(x4, start_dim=2)
        x=x.permute(0,2,1)
        x5=self.atten(x,x,x)
        x=torch.flatten(x5[0],start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder_fc(z)
        x = x.view(-1, 128, 4, 4)
        x = self.upconv1(torch.cat([x, x4], dim=1))
        x = self.upconv2(torch.cat([x, x3], dim=1))
        x = self.upconv3(torch.cat([x, x2], dim=1))
        x = self.upconv4(torch.cat([x, x1], dim=1))
        return x, mu, logvar,x5

    def loss_function(self,recons,input,mu,log_var):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        kld_weight = 1
        recons_loss =F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples
        


class VAEs(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAEs, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()  # 输出均值和log方差
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )


class DualVAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DualVAE, self).__init__()
        self.atten = nn.MultiheadAttention(256,4)
        self.fc_mu = nn.Linear(256, latent_dim*2)  # 假设图像尺寸缩小到 4x4
        self.fc_logvar = nn.Linear(256, latent_dim*2)
        self.image_vae = VAEs(input_dim, latent_dim)
        self.text_vae = VAEs(input_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        
        return mu + eps * std
    def forward(self, image_emb, text_emb):
        h1=self.image_vae.encoder(image_emb)
        h2=self.text_vae.encoder(text_emb)
        h=torch.cat((h1,h2),dim=-1)
        h0=self.atten(h,h,h)
        mu = self.fc_mu(h0[0])
        logvar = self.fc_logvar(h0[0])
        z = self.reparameterize(mu, logvar)
        z_i,z_t = torch.chunk(z, 2, dim=-1)
        recon_image = self.image_vae.decoder(z_i)
        recon_text = self.text_vae.decoder(z_t)
        return recon_image, recon_text, mu, logvar, h0[0]
    
    def contrastive_loss(self, zt, zi, delta=0.1):
        """
        计算对比损失
        :param t: 文本嵌入 (batch_size, embed_dim)
        :param v: 图像嵌入 (batch_size, embed_dim)
        :param delta: 超参数
        :return: 对比损失
        """
        batch_size = 64
        
        # 定义函数 h 和 g

        # 计算相似度矩阵
        sim_matrix = torch.matmul(zt, zi.t())  # (batch_size, batch_size)
        
        # 计算对比损失
        loss = 0.0
        epsilon = 1e-8  # 小常数，防止分母为零
        for i in range(batch_size):
            pos_sim = sim_matrix[i, i] - delta
            neg_sim_t = sim_matrix[i, :]  # 所有与 t_i 的相似度
            neg_sim_v = sim_matrix[:, i]  # 所有与 v_i 的相似度
            
            # 去掉对角线上的正样本相似度
            neg_sim_t = torch.cat([neg_sim_t[:i], neg_sim_t[i+1:]])
            neg_sim_v = torch.cat([neg_sim_v[:i], neg_sim_v[i+1:]])
            
            # 计算损失的两部分
            loss_t = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim_t)) + epsilon))
            loss_v = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim_v)) + epsilon))
            
            loss += loss_t + loss_v
            
        return loss / batch_size

    def loss_function(self, recon_image,  recon_text,image_emb,text_emb, mu, logvar, latent_emb, weights):
        recon_loss_image = F.mse_loss(recon_image, image_emb, reduction='sum')
        recon_loss_text = F.mse_loss(recon_text, text_emb, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        zi,zt=torch.chunk(latent_emb, 2, dim=-1)
        contrastive_loss = self.contrastive_loss(zi,zt)
        # contrastive_loss = F.cosine_embedding_loss(mu_image, mu_text, torch.ones(mu_image.size(0)).to(mu_image.device))
        total_loss = (weights['recon_image'] * recon_loss_image + 
                  weights['recon_text'] * recon_loss_text + 
                  weights['kl'] * kl_loss +
                  weights['contrastive'] * contrastive_loss)
        return total_loss, recon_loss_image , recon_loss_text , kl_loss , contrastive_loss




class VLVAE(nn.Module):
    def __init__(self,latent_dim: int,):
        super(VLVAE, self).__init__()


        # 编码器
        self.convblk1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=7, stride=2, padding=3), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),)
        self.convblk2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),)
        self.convblk3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),)
        self.convblk4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
        )
        self.atten = nn.MultiheadAttention(128,4)

        # 潜在空间的线性变换
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)  # 假设图像尺寸缩小到 4x4
        self.fc_logvar = nn.Linear(128 * 4 * 4, latent_dim)

        # 解码器
        self.decoder_fc = nn.Linear(latent_dim, 128 * 4 * 4)
        
        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),)
        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),)
        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),)
        self.upconv4 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x1=self.convblk1(x)
        x2=self.convblk2(x1)
        x3=self.convblk3(x2)
        x4=self.convblk4(x3)

        x = torch.flatten(x4, start_dim=2)
        x=x.permute(0,2,1)
        x5=self.atten(x,x,x)
        x=torch.flatten(x2[0],start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        x = self.decoder_fc(z)
        x = x.view(-1, 128, 4, 4)

        x = self.upconv1(torch.cat([x, x4], dim=1))
        x = self.upconv2(torch.cat([x, x3], dim=1))
        x = self.upconv3(torch.cat([x, x2], dim=1))
        x = self.upconv4(torch.cat([x, x1], dim=1))


        return x, mu, logvar,x5
    
    def forward_l(self, x):
        pass
    def loss_function(self,recons,input,mu,log_var):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        kld_weight = 1
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def cons_loss(self, h_t, g_a):
        B = h_t.size(0)
        
        consM = torch.exp(h_t@g_a.T)
        positive_term =torch.diag(consM- self.delta)
        neg_term1 = torch.sum(consM,dim=0)
        neg_term2 = torch.sum(consM,dim=1)
        
        # Calculate the log terms
        log_term_1 = torch.log(positive_term / (positive_term + neg_term1))
        log_term_2 = torch.log(positive_term / (positive_term + neg_term2))
        
        # Calculate the final loss
        loss = -torch.mean(log_term_1 + log_term_2)
        
        return loss
    

    def sample(self,
               num_samples:int,
               current_device: int):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples
        