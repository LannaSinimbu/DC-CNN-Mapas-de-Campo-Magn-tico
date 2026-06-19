# Pasta de dados

Coloque aqui os mapas de campo magnético medidos, em formato de texto
(`np.loadtxt` lê uma matriz 2D de valores separados por espaço).

Os arquivos esperados pela configuração padrão (`configs/config.yaml`) são:

- `MAPx260620241658.txt` (z = 200 µm)
- `MAPx270620240930.txt` (z = 400 µm)
- `MAPx020720241338.txt` (z = 670 µm)
- `MAPx030720241005.txt` (z = 720 µm)
- `MAPx040720240950.txt` (z = 900 µm)

Edite `configs/config.yaml` para usar seus próprios arquivos e alturas.

> Os arquivos `.txt` são ignorados pelo Git (`.gitignore`) para não versionar
> dados experimentais grandes. Se quiser distribuir os dados, considere um
> repositório de dados separado (ex.: Zenodo) e cite o DOI aqui.
