import torch
class Agent():
    def __init__(self,
                actor,
                obs_dim,
                action_dim,
                lr = 0.001,
                epsilon = 0.1):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.actor = actor
        self.target_actor = copy.deepcopy(actor)
        self.lr = lr
        self.epsilon = epsilon
        self.global_step = 0
        # every 200 training steps, coppy model param into target_model
        self.update_target_steps = 200  
        self.optimizer_actor = torch.optim.Adam(actor.parameters(), lr=lr)
    
    def sample(self, obs):
        obs = torch.from_numpy(obs.astype(np.float32)).view(1,-1)
        # choose action based on prob
        action = np.random.choice(range(self.action_dim), p=self.target_actor(obs).detach().numpy().reshape(-1,))  
        return action

    def predict(self, obs):  # choose best action
        obs = torch.from_numpy(obs.astype(np.float32)).view(1,-1)
        return self.actor(obs).argmax(dim=1).item()

    def sync_target(self):
        self.target_actor.load_state_dict(self.actor.state_dict())

    def learn(self, batch_obs, batch_action, batch_adv):
        # update target model
        if self.global_step % self.update_target_steps == 0:
            self.sync_target()
        self.global_step += 1
        self.global_step %= 200

        pred_action_distribution = self.actor(batch_obs)
        target_action_distribution = self.target_actor(batch_obs).detach()
        true_action_distribution = nn.functional.one_hot(batch_action, num_classes=self.action_dim)
        pred_choosed_action_prob = (pred_action_distribution*true_action_distribution).sum(1, keepdim=True)
        target_choosed_action_prob = (target_action_distribution*true_action_distribution).sum(1,keepdim=True)
        J = (pred_choosed_action_prob/target_choosed_action_prob*batch_adv.view(-1,1))
        J_clip = (torch.clamp(pred_choosed_action_prob/target_choosed_action_prob, 1-self.epsilon, 1+self.epsilon)*batch_adv.view(-1,1))
        self.optimizer_actor.zero_grad()
        loss = -1.0*(torch.cat((J, J_clip), dim=1).min(dim=1)[0]).mean()
        loss.backward()
        self.optimizer_actor.step()
        return loss