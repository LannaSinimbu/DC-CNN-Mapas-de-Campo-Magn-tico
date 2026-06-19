# DC-CNN: Continuação para Baixo + CNN para Mapas de Campo Magnético

Pipeline para refinamento e reconstrução de mapas de campo magnético
$B_z(x, y)$ medidos por magnetometria de varredura. Combina **continuação
para baixo** (*downward continuation*) e **deconvolução do footprint do
sensor** no domínio de Fourier com uma **U-Net 2D** treinada para mapear os
campos continuados a um plano de referência mais próximo da amostra.

## Visão geral do método

1. **Carregamento** dos mapas medidos em diferentes alturas sensor–amostra.
2. **Operações físicas** (domínio de Fourier):
   - continuação para baixo regularizada — leva o campo de uma altura
     $z_{\text{from}}$ a uma altura menor $z_{\text{to}}$ via fator
     $e^{k\,\Delta z}$ com regularização de Tikhonov;
   - deconvolução de um *footprint* gaussiano do sensor.
3. **Aumento de dados** fisicamente seguro (flips, rotação de 180°, jitter de
   fundo planar, ruído gaussiano).
4. **Construção de pares** entrada → alvo, onde a entrada é o mapa continuado
   + deconvoluído e o alvo é o mapa de referência (menor altura).
5. **Treinamento** de uma U-Net 2D em patches sobrepostos (perda L1).
6. **Predição**: gera um mapa em uma nova altura a partir de um mapa medido.

## Estrutura do projeto

```
dc-cnn-bz-maps/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── configs/
│   └── config.yaml          # caminhos, alturas e hiperparâmetros
├── data/
│   └── README.md            # onde colocar os mapas .txt
├── notebooks/
│   └── model_DC_CNN.ipynb   # notebook original (exploração)
├── outputs/                 # modelos treinados e mapas gerados
└── src/
    ├── __init__.py
    ├── physics.py           # k-grid, z0, downward continuation, deconvolução
    ├── data.py              # carregamento, augmentation, pares, patches, Dataset
    ├── model.py             # U-Net 2D
    ├── config.py            # leitura do YAML
    ├── train.py             # script de treino (CLI)
    └── predict.py           # script de predição (CLI)
```

## Instalação

```bash
git clone https://github.com/<SEU_USUARIO>/dc-cnn-bz-maps.git
cd dc-cnn-bz-maps

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Dados

Coloque os mapas `.txt` na pasta `data/` e ajuste `configs/config.yaml` com os
nomes dos arquivos e as alturas correspondentes (em micrômetros). Veja
[`data/README.md`](data/README.md). Os `.txt` são ignorados pelo Git por
padrão para não versionar dados experimentais grandes.

## Uso

### Treinar

```bash
python -m src.train --config configs/config.yaml
```

O modelo treinado é salvo em `outputs/modelo_dc_cnn.pth` (configurável).

### Gerar um mapa em nova altura

```bash
python -m src.predict \
    --config configs/config.yaml \
    --input data/MAPx260620241658.txt \
    --z-from 200 --z-to 140 \
    --output outputs/Bz_140um_CNN.txt \
    --plot
```

`--z-from` e `--z-to` em micrômetros; `--plot` mostra a comparação
medido / DC+deconv / CNN.

### Usar como biblioteca

```python
import numpy as np
from src import downward_continuation, get_k_grid, UNet2D

bz = np.loadtxt("data/MAPx260620241658.txt")
ny, nx = bz.shape
_, _, k = get_k_grid(nx, ny, dx=40e-6, dy=40e-6)
bz_140 = downward_continuation(bz, z_from=200e-6, z_to=140e-6, k=k)
```

## Configuração

Todos os parâmetros ficam em `configs/config.yaml`: passos espaciais
(`dx`, `dy`), regularizações (`alpha_dc`, `beta`), largura do footprint
(`sigma_psf`), tamanho de patch, taxa de aprendizado, número de épocas etc.

## Observações sobre o método

- A estimativa de `z0` em `estimate_z0` é heurística (inclinação do espectro
  radial em escala log). Use-a como referência inicial, não como medida
  física rigorosa da altura de voo.
- A continuação para baixo amplifica altas frequências; o termo `alpha_dc`
  controla esse efeito e deve ser ajustado conforme o nível de ruído.
- A U-Net exige dimensões múltiplas de 8 (três níveis de *pooling*); a
  predição faz o recorte automaticamente (`crop_to_mult8`).

## Citação

Se este código for útil em uma publicação, considere citar o trabalho
associado. Adicione aqui a referência / DOI quando disponível.

## Licença

[MIT](LICENSE). Edite o campo de copyright com seu nome.
