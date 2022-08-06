# Collaborative Inference of Diffusion Models

This is an example of collaborative inference of the latent diffusion model from [this notebook](https://colab.research.google.com/github/multimodalart/latent-diffusion-notebook/blob/main/Latent_Diffusion_LAION_400M_model_text_to_image.ipynb) by [@multimodalart](https://twitter.com/multimodalart).

**Idea:** A swarm of servers from all over the Internet hold a model on their GPUs and respond to clients' queries to run inference.
The queries are evenly distributed among all servers connected to the swarm. Any GPU owner who is willing to help may
run a server and connect to the swarm, thus increasing the total system throughput.

- Model: [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)
- Dataset: [LAION-400M](https://laion.ai/laion-400-open-dataset/)
- Distributed inference: [hivemind](https://github.com/learning-at-home/hivemind)

**Warning:** This is a demo for research purposes only. Some safety features of the original model may be disabled.

## Installing requirements
```bash

conda create -y --name demo-for-laion python=3.8.12 pip
conda activate demo-for-laion
conda install -y -c conda-forge cudatoolkit-dev==11.3.1 cudatoolkit==11.3.1 cudnn==8.2.1.32
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install https://github.com/learning-at-home/hivemind/archive/61e5e8c1f33dd2390e6d0d0221e2de6e75741a9c.zip
pip install matplotlib
```

## How to call remote inference

Call the remote inference (no need to have a GPU):

```python
from diffusion_client import DiffusionClient

# Here, you can specify one or more addresses of any servers
# connected to the swarm (no need to list all of them)
client = DiffusionClient(initial_peers=[
    '/ip4/193.106.95.184/tcp/31234/p2p/Qmas1tApYHyNWXAMoJ9pxkAWBXcy4z11yquoAM3eiF1E86',
    '/ip4/193.106.95.184/tcp/31235/p2p/QmYN4gEa3uGVcxqjMznr5vEG7DUBGUWZgT98Rnrs6GU4Hn',
])

images = client.draw(2 * ['a photo of the san francisco golden gate bridge',
                          'graphite sketch of a gothic cathedral',
                          'hedgehog sleeping near a laptop'])
```

Draw results:

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for index, img in enumerate(images):
    plt.subplot(2, 3, index + 1)
    plt.imshow(img)
    plt.axis('off')
plt.tight_layout()
plt.show()
```

**[[Try it in Google Colab]](https://colab.research.google.com/drive/1_XtEjXzskKRrgPcvCYXjLu2g34jZE0Zo?usp=sharing)**

Expected output:

<img src="https://github.com/learning-at-home/demo-for-laion/blob/main/img/example_output.png" width="560">

## How to run a new server

First, you need to install all dependencies for the model from the original [Colab notebook](https://colab.research.google.com/github/multimodalart/latent-diffusion-notebook/blob/main/Latent_Diffusion_LAION_400M_model_text_to_image.ipynb). Then, you can run:

```python
python -m run_server --identity server1.id --host_maddrs "/ip4/0.0.0.0/tcp/31234" --initial_peers \
    "/ip4/193.106.95.184/tcp/31234/p2p/Qmas1tApYHyNWXAMoJ9pxkAWBXcy4z11yquoAM3eiF1E86" \
    "/ip4/193.106.95.184/tcp/31235/p2p/QmYN4gEa3uGVcxqjMznr5vEG7DUBGUWZgT98Rnrs6GU4Hn"
# Skip --initial_peers if you'd like to start a new swarm
```

Ensure that `--max-batch-size` is small enough for your GPU to do inference without running out of memory. The default value is 16.

If your public IP address doesn't match the IP address of the network interface, use `--announce_maddrs /ip4/1.2.3.4/tcp/31324`
to announce your public IP to the rest of the network.

## Authors

[
Based on assorted code by shuf([mryab@](https://github.com/mryab) [younesbelkada@](https://github.com/younesbelkada) [borzunov@](https://github.com/borzunov) [timdettmers@](https://github.com/timdettmers) [dbaranchuk@](https://github.com/dbaranchuk) [greenfatguy@](https://github.com/GreenFatGuy) [artek0chumak@](https://github.com/artek0chumak) and [hivemind](https://github.com/learning-at-home/hivemind) contributors)
]
