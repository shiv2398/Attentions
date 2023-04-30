class Attention(nn.Module):
    
    def __init__(self,hidden_dim,input_dim=None,proj_values=False):
        
        super().__init__()
        self.d_k=hidden_dim
        self.input_dim=hidden_dim if input_dim is None else input_dim
        
        self.proj_values=proj_values
        
        self.linear_query=nn.Linear(self.input_dim,hidden_dim)
        self.linear_key=nn.Linear(self.input_dim,hidden_dim)
        self.linear_value=nn.Linear(self.input_dim,hidden_dim)
        
        self.alphas=None
        
    def init_keys(self, keys):
        #print('\n\nkey Intialisation')
        self.keys = keys
        #print(f'\nkeys : {self.keys}')
        self.proj_keys = self.linear_key(self.keys)
        
        self.values = self.linear_value(self.keys) if self.proj_values else self.keys
        #print(f'\nValues : {self.values}')
    def scores_function(self,query):
        
        proj_query=self.linear_query(query)
        
        dot_products=torch.bmm(proj_query,self.proj_keys.permute(0,2,1))
        #print(f'\nQ . K : {dot_products}')
        scores=dot_products/np.sqrt(self.d_k)
        #print(f'\nScores :{scores}')
        return scores
    
    def forward(self,query,mask=None):
        scores=self.scores_function(query)
        if mask is not None:
            scores=scores.masked_fill(mask==0,-1e9)
        #print(f'\nMask : {mask}')
        alphas=F.softmax(scores,dim=-1)
        self.alphas=alphas.detach()
    
        context=torch.bmm(alphas,self.values)
        return context
    
        
        
