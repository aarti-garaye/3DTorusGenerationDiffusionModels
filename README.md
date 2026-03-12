# 3DTorusGenerationDiffusionModels
Extending the swiss roll example for a 3D torus generation using multi rate diffusion models (one for the cordinatews for the torus and other for the color based on the curvature)

# Original Work

Asan Medical Center 
## Reverse Process (noise -> data)

$$p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^{T}p_{\theta}(x_{t-1} \vert x_t),\ \ \ \text{where}\ \ \ p_{\theta}(x_{t-1} \vert x_t) = \mathcal{N}(x_{t-1} ; \mu_{\theta}(x_t, t), \Sigma_\theta(x_t, t))$$
<br>
<br>
## Forward Process (data -> noise)
$$ q(x_{1:T} \vert x_0) = \prod_{t=1}^Tq(x_t \vert x_{t-1}),\ \ \ \text{where}\ \ \ q(x_t \vert x_{t-1})=\mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I})$$

<br>
<br>

Let $\alpha_t = 1-\beta_t$ and $\bar{\alpha_t}=\prod_{s=1}^t\alpha_s$, then $q(x_t \vert x_0)=\mathcal{N}(x_t ; \sqrt{\bar{\alpha_t}}x_0,(1-\bar{\alpha_t})\mathbf{I})$

<br>
<br>

$$ \mathbb{E}_q[-\log p_\theta(x_0)] \leq \mathbb{E}_q[-\log {{p_\theta(x_{0:T})}\over{q(x_{1:T} \vert x_0)}}]=\mathbb{E}_q[-\log p(x_T)-\sum_{t\geq1} \log {{ {p_\theta(x_{t-1} \vert x_t)} }\over{q(x_t \vert x_{t-1})}} ]=L $$

<br>
<br>

Rewrite $L$:
$$\begin{align}L &= \mathbb{E}_q [ D_{KL}(q(x_T|x_0) || p(x_T)) + \sum_{t>1}D_{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t))-\log p_\theta (x_0|x_1) ] \end{align}$$
where, $q(x_{t-1} \vert x_t, x_0)=\mathcal{N}(x_{t-1};\tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t)$ with $\tilde{\mu}_t(x_t, x_0)=\frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1-\bar{\alpha}_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1}) }{1-\bar{\alpha}_t}x_t$ and $\tilde{\beta}_t=\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t}\beta_t$

<br>
<br>

We fixed $\beta_t$ in Forward Process. And $ \Sigma_\theta(x_t,t)=\sigma_t^2$ set $\beta_t$ or ${{ 1-\bar{\alpha}_{t-1} }\over{1-\bar{\alpha}_{t}}}\beta_t$ and the $\mu_{\theta}$ is:
$$\mu_{\theta}(x_t, t) \approx \tilde{\mu}_t(x_t, x_0) =\frac{1}{\sqrt{\alpha_t}}(x_t(x_0, \epsilon) - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon)$$

<br>
<br>

Let $\epsilon_{\theta}$ be a approximator, which predicts a $\epsilon$ from $x_t$. Then, the $\mu_{\theta}$ is:
$$\mu_{\theta}(x_t, t) =\frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_{\theta}(x_t, t))$$

<br>
<br>

Therefore, to sample $x_{t-1} \sim p_{\theta}(x_{t-1} \vert x_t)$ is to compute $x_{t-1} = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_{\theta}(x_t, t)) + \sigma_tz $

<br>
<br>

Finally, since $x_t \sim q(x_t \vert x_0)=\mathcal{N}(x_t ; \sqrt{\bar{\alpha_t}}x_0,(1-\bar{\alpha_t})\mathbf{I})$, the training objective function is:

$$\mathcal{L}_{simple}(\theta)=\mathbb{E}_{t, x_0, \epsilon}[\Vert \epsilon - \epsilon_{\theta}(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon\Vert^2]$$

# Enhancements to the Notebook

I plan on building upon the existing notebook by changing the Swiss roll into a 3D Torus. Part 1 of the enhancements will be generating a torus in a 3D space, calculating the local curvature at each point, and mapping the curvature to color *(RGB or single channel)*

Part 2 will be Multi Rate Diffusion, I will modify the diffusion process so the position (x,y,z coordinates of the torus) and the color denoises at different speeds. I think I would like the position to denois faster than color.

I plan on adding some visualization like in the notebook above, maybe a 3D scatterplot to show the denoising process and then a paragraph explaining my extensions

## Generating Torus

A torus is like a donut so we can think of it's generation as a circle of radius $r$ is revolving around an axis at some distance, $R$, from the circle's center. According to Gemini, the mathematical/programming generation of torus is by generating a mesh by iterating through two angels, $\theta$ (toroidal angle), and $\phi$ (poloidal angle).

$$x=(R + r\cos\phi)\cos\theta \\ y = (R + r\cos\phi)\sin\theta \\ z = r\sin\phi$$ Note that both $$\theta, \phi \in [0, 2\pi]$$.

We also need the curvature formula,
$$K(\phi) = \frac{\cos\phi}{r(R+r\cos\phi)}$$
The curvature varies smoothly across the torus, the outer edge looks like a sphere and it positively curved, the inner edge (donut hole) is a saddle like shape negatively curved. And the top of the torus is like a cylinder.

## Multi-Rate Diffusion
Instead of the one standard $\beta_t$ schedule, we use two different schedule to add noise.

- Fast schedule for position (x,y,z). This will denoise quickly
- Slow schedule for color, this will take longer to denoise.

By doing this I am testing whether spatial structure and appearance information benefit from two different diffusion dynamics. Note that both are predicted using the same nueral network

# Summary of Extension / Enhancements

I extended the 2D Swiss roll to a 3D torus with curvature based coloring. I also used two different diffusion rate i.e. different noise schedule for the color and the coordinates of the torus. I tried to use the same hyperparameters as the Swiss Roll to compare the results. I used PyTorch models with 500 epochs for training, this took around 5-7 minutes for one million points.

## Results

- The model learned to generate torus in a three dimensional space
- The generated samples from the denoising process had realistic curvature patterns as well as the x, y, z coordinate positions
- The results of different schedules for adding the noise is immidiately clear from the model but the annimation shows that the structure of the torus emerges quicker than the curvature (color). But they both converged to be a realistic torus.

## Comments and Future Direction
I used 500 epochs for training the model, looking at the loss function it seems like we didn't change the MSE between 350-500 epochs, but we had to wait almost 2-4 more minutes for the model to train on 500. I think we would get similar results if we trained the model on 350 or even 300 epochs.

Even though the color (curvature) and position (coordinates) of the torus were on different schedule for adding the noise, both were trained on the same nueral network. This is the same one used for the original Swiss Roll example. Furthermore, it uses the same Backbone and DiffusionProcess class as the original examples in the notebook.

I also had to add a Gaussian noise while generating the torus because it was 3D. In the Swiss Roll example, the DiffusionProcess class took care of it, but for the torus I needed to make sure that it had density everywhere in $\mathbb{R}^3$. If we removed this then the torus becomes purely 2D surface in a 3D space. If we did this then the diffusion model will have a much harder learning problem because the data distribution is no longer a proper density in $\mathbb{R}^3$.
