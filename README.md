# Reconhecimento facial em tempo real

Reconhecimento facial por webcam com a stack atual de pesquisa: detecção **YuNet**, embeddings **ArcFace** em ONNX, controle de **qualidade** (eDifFIQA), **detecção de vivacidade** (MiniFASNet + desafio ativo), **rastreamento com votação**, **limiar calibrado**, galeria **cifrada** e identificação sobre templates cifrados com **IDFace** (Paillier).

Roda igual em Windows, Linux e macOS, sem compilar nada (sem dlib, CMake ou MinGW).

```
webcam ─► YuNet (rosto + 5 pontos) ─► alinhamento 112×112 ─► qualidade (FIQA)
              │                                                   │ aprovado
              └──► rastreador IoU ◄── votação ◄── identificação ◄── ArcFace (512-d)
                                                  ▲          ▲      + MiniFASNet (vivo?)
                                       galeria AES-GCM   ou   IDFace (Paillier)
                                                             └─► servidor de chave
```

## O que mudou em relação à versão com `face_recognition`

| | Antes | Agora |
|---|---|---|
| Detecção | HOG do dlib (só rosto frontal e grande) | YuNet (rostos pequenos, de perfil, pouca luz) + alinhamento por 5 pontos |
| Reconhecimento | dlib 128-d (~2017) | ArcFace R50 512-d treinado no WebFace600K (ou SFace/AdaFace) |
| Decisão | tolerância fixa 0,6 | limiar calibrado com os seus dados para uma FAR alvo |
| Qualidade | nenhuma | eDifFIQA + tamanho, nitidez e ângulo; cadastro mais exigente |
| Ataque com foto/tela | aceito | MiniFASNet (passivo) + desafio de virar o rosto (ativo) |
| Estabilidade | reconhece todo frame, nome "pisca" | rastreamento + votação; reconhece a cada N frames |
| Cadastro | 1 foto, nome pelo arquivo | várias fotos ou webcam, média dos embeddings, descarta fotos de outra pessoa |
| Armazenamento | `pickle` (executa código se adulterado), em claro, no Git | SQLite com AES-256-GCM, fora do Git; ou IDFace (só ciphertexts) |
| LGPD | — | consentimento registrado, retenção, expurgo, eliminação a pedido |
| Instalação | compilar dlib (~7 min, CMake, MinGW) | `pip install` de wheels prontos |

## Instalação

Requer **Python 3.10+**.

```bash
cd FaceRecognition/face_recognizer
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python baixar_modelos.py           # ~190 MB, com verificação SHA-256
python teste_webcam.py             # confere se a câmera abre
```

No Linux, para acessar a webcam: `sudo usermod -aG video $USER` (depois saia e entre de novo na sessão).

## Ordem para testes

```bash
cd FaceRecognition/face_recognizer
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python baixar_modelos.py
python -m pip install pytest && python -m pytest tests      # devem passar 27 testes
python teste_webcam.py
```

### Modelos e licenças

| Modelo | Uso | Licença |
|---|---|---|
| YuNet (OpenCV Zoo) | detecção + 5 pontos | MIT |
| ArcFace R50 `w600k_r50` (InsightFace buffalo_l) | reconhecimento (padrão) | **somente pesquisa não comercial** |
| SFace (OpenCV Zoo) | reconhecimento alternativo | Apache-2.0 |
| eDifFIQA tiny (OpenCV Zoo) | qualidade da imagem | CC-BY-4.0 |
| MiniFASNetV2-SE (facenox) | vivacidade passiva | Apache-2.0 |

O ArcFace serve para o projeto acadêmico. **Para uso comercial**, defina `FACEREC_MODELO=sface` antes de cadastrar. Outra opção é usar um AdaFace (MIT) exportado para ONNX: `FACEREC_MODELO=onnx:/caminho/adaface.onnx:bgr`.

## Uso

### 1. Cadastro

Uma pasta por pessoa (o formato antigo `images/Nome_1.jpg` também funciona). Use de 3 a 10 fotos de frente, bem iluminadas, com pequenas variações:

```
images/Maria/1.jpg  images/Maria/2.jpg  images/Joao/1.jpg ...
```

```bash
python cadastrar.py --consentimento "termo assinado em 28/10/2026"
python cadastrar.py --webcam "Maria" --consentimento "termo assinado em 28/10/2026"   # direto da câmera
python cadastrar.py --consentimento "..." --apagar-fotos   # apaga as fotos depois (minimização)
```

Cada foto passa pelo controle de qualidade, e a pessoa vira um único template: a média normalizada das fotos aprovadas. Recadastrar o mesmo nome substitui o cadastro anterior.

### 2. Reconhecimento

```bash
python reconhecer.py                 # webcam
python reconhecer.py --desafio       # exige virar o rosto na ordem sorteada antes de liberar
python reconhecer.py --video arquivo.mp4 --sem-janela --saida anotado.mp4
```

Rótulos possíveis: nome + similaridade, `Desconhecido`, `verificando...`, o motivo de qualidade (ex.: `rosto muito de lado`) ou `ATAQUE?` quando a vivacidade média da trilha fica abaixo de 50%.

### 3. Calibração do limiar

O limiar padrão (0,40 de cosseno para ArcFace) é conservador. Para medir o seu, grave vídeos curtos (10–20 s) de várias pessoas com a sua câmera e rode:

```bash
python calibrar.py --video Maria=maria.mp4 --video Joao=joao.mp4 --video Ana=ana.mp4 --far 0.001
```

O script simula a operação real (parte das amostras vira template e o resto vira consulta), mede genuínos e impostores e salva em `output/calibracao.json` o limiar que mantém a taxa de falsa aceitação no alvo, além do τ do IDFace para cada β. Para estimar FAR = 0,1% com confiança são necessários pelo menos 10 000 pares de impostor. Com poucas pessoas, o script avisa e usa uma margem de segurança.

### 4. Gerenciamento (direitos do titular)

```bash
python gerenciar.py listar
python gerenciar.py remover --nome Maria
python gerenciar.py expurgar          # remove cadastros com prazo de retenção vencido
```

## IDFace: identificação sobre templates cifrados

Implementação da variante Paillier de *"IDFace: Face Template Protection for Efficient and Secure Identification"* (arXiv:2507.12050, 2025). Templates faciais não são senhas: não dá para trocá-los se vazarem, e já foi demonstrado que é possível reconstruir o rosto a partir deles. No modo IDFace, **o servidor local nunca guarda um template decifrável**: ele compara a consulta com a base cifrada, e só um segundo servidor (o de chave) consegue decifrar as pontuações.

Como funciona:
1. **Transformação quase isométrica**: o template de 512 dimensões vira um vetor ternário com as α=341 maiores componentes trocadas pelo sinal (±1). A consulta usa β ∈ {63, 127, 341} componentes. A pontuação inteira acompanha o cosseno (r ≥ 0,98).
2. **x = x⁺ − x⁻**: dois vetores binários, então ⟨x,y⟩ = ⟨x⁺,y⁺⟩ + ⟨x⁻,y⁻⟩ − ⟨x⁺,y⁻⟩ − ⟨x⁻,y⁺⟩.
3. **Empacotamento**: 292 pessoas por ciphertext (β_max=127, base 128, Paillier de 2048 bits), como dígitos de um número grande.
4. **Produto interno cifrado só com multiplicações de ciphertexts**: como a consulta é ternária, não há nenhuma exponenciação na consulta.
5. **Dois servidores**: o local tem a base cifrada e a chave pública; o de chave tem só a chave secreta, decifra e responde apenas *qual posição* passou do limiar τ.

Além do artigo, esta implementação acrescenta:
- **eliminação real** (LGPD art. 18): a contribuição da pessoa é subtraída homomorficamente do bloco, usando um token de remoção de uso único registrado no servidor de chave;
- **recifragem completa a cada mudança**, para que comparar versões da base não revele quais dimensões mudaram;
- **cifragem DJN** e paralelismo (como a Intel IPCL usada no artigo);
- no servidor de chave: tokens de acesso separados (identificação e administração), limite de requisições e log de auditoria sem pontuações, para que ele não vire um oráculo de decifração.

### Como rodar

```bash
# no servidor de chave (em produção, outra máquina)
export FACEREC_IDFACE_SENHA='senha da chave'  FACEREC_IDFACE_TOKEN='...'  FACEREC_IDFACE_ADMIN='...'
python idface_servidor_chave.py gerar-chaves      # gera a chave e output/idface_chave_publica.json
python idface_servidor_chave.py servir            # http://127.0.0.1:8765

# no servidor local (copie para ele só o idface_chave_publica.json)
export FACEREC_IDFACE_URL=http://servidor-chave:8765  FACEREC_IDFACE_TOKEN='...'  FACEREC_IDFACE_ADMIN='...'
python cadastrar.py --protecao idface --consentimento "..."
python reconhecer.py
# depois de calibrar: no servidor de chave, python idface_servidor_chave.py importar-calibracao
```

Fora de `localhost`, o servidor de chave só sobe com os dois tokens definidos. Coloque-o atrás de TLS (nginx, Caddy).

### Desempenho medido (CPU de 2 núcleos, Python + gmpy2)

| Pessoas | Identificação β=63 | Identificação β=127 | Cadastro em lote | Base cifrada | Busca sem proteção |
|---|---|---|---|---|---|
| 1 000 | 29 ms | 28 ms | 17 s | 2,7 MB | 0,7 ms |
| 10 000 | 238 ms | 285 ms | 165 s | 24 MB | 3,8 ms |

Cerca de 80% do tempo de identificação é a decifração no servidor de chave, e o custo cresce linearmente com o número de blocos de 292 pessoas. O cadastro escala com os núcleos da CPU. Com o rastreamento, cada pessoa na frente da câmera é reidentificada só a cada 5 frames, então 10 000 cadastros ainda funcionam em tempo real.

Precisão: nos testes deste projeto, a decisão do IDFace concordou com a do cosseno em 98,6–99,2% dos pares. O EER foi de 0,61% em texto claro contra 0,69–0,75% com IDFace, a mesma ordem da perda "menor que 1%" do artigo. As pontuações cifradas são **idênticas** às calculadas em claro com os vetores ternários (verificado nos testes).

### Limitações (mesmo modelo de ameaça do artigo)

- O IDFace protege contra quem obtém a **base cifrada** (roubo de disco ou backup). O servidor local vê a consulta e o template no momento do cadastro, como no artigo.
- O servidor de chave vê as pontuações decifradas da consulta contra todos os cadastrados (não vê os templates) e vê o template de quem está sendo removido. Ele precisa ser operado por outra pessoa ou máquina.
- Quem tiver a base **e** acesso autenticado ao servidor de chave pode fazer consultas. Por isso existem o limite de taxa e a auditoria.
- A variante CKKS do artigo (cerca de 50× mais rápida para milhões de pessoas) não foi implementada: Paillier é suficiente para milhares de pessoas.

## LGPD

Biometria facial é **dado pessoal sensível** (art. 5º, II). O sistema foi preparado para as boas práticas, mas a conformidade depende de como ele é usado:

- **Base legal (art. 11)**: consentimento específico e destacado, ou prevenção à fraude e segurança do titular. Legítimo interesse não serve. O `--consentimento` registra como o consentimento foi obtido.
- **Alternativa**: ofereça outra forma de acesso a quem não quiser usar biometria.
- **Minimização**: só o template é guardado, nunca a foto (`--apagar-fotos`).
- **Segurança**: AES-256-GCM em repouso, chave em arquivo `0600` ou na variável `FACEREC_CHAVE_GALERIA`, ou IDFace.
- **Retenção**: `FACEREC_RETENCAO_DIAS` (padrão 365) e `gerenciar.py expurgar`.
- **Direitos do titular**: `gerenciar.py listar` (acesso) e `remover` (eliminação, inclusive no IDFace).
- **RIPD** (relatório de impacto): recomendado para uso em larga escala ou em locais como condomínios.
- `images/`, `output/` e `models/` estão no `.gitignore`: **nunca** faça commit de fotos, templates ou chaves.

## Testes

```bash
python -m pip install pytest
python -m pytest tests
```

São 27 testes: Paillier, correção exata das pontuações cifradas do IDFace contra o texto claro, remoção e reuso de slot, proteção contra oráculo, HTTP com autenticação, galeria cifrada (adulteração, troca de linhas, retenção), rastreamento, desafio ativo, alinhamento e embeddings.

## Configuração

Todos os parâmetros ficam em `facerec/config.py`. As variáveis de ambiente principais são:

| Variável | Para que serve |
|---|---|
| `FACEREC_MODELO` | `arcface`, `sface` ou `onnx:caminho[:bgr]` |
| `FACEREC_MODELOS` | pasta dos modelos |
| `FACEREC_SAIDA` | pasta da galeria e da calibração |
| `FACEREC_CHAVE_GALERIA` | chave AES em base64 |
| `FACEREC_RETENCAO_DIAS` | prazo de retenção dos cadastros |
| `FACEREC_IDFACE_URL` | endereço do servidor de chave |
| `FACEREC_IDFACE_TOKEN`, `FACEREC_IDFACE_ADMIN` | tokens de acesso ao servidor de chave |
| `FACEREC_IDFACE_SENHA` | senha da chave secreta |

## Referências

- Deng et al., *ArcFace: Additive Angular Margin Loss for Deep Face Recognition*, CVPR 2019.
- Kim et al., *AdaFace: Quality Adaptive Margin for Face Recognition*, CVPR 2022.
- Wu et al., *YuNet: A Tiny Millisecond-level Face Detector*, 2023.
- Babnik et al., *eDifFIQA: Towards Efficient Face Image Quality Assessment based on Denoising Diffusion Probabilistic Models*, IEEE T-BIOM 2024.
- *IDFace: Face Template Protection for Efficient and Secure Identification*, arXiv:2507.12050, 2025.
- Paillier, *Public-Key Cryptosystems Based on Composite Degree Residuosity Classes*, EUROCRYPT 1999; Damgård, Jurik e Nielsen, *A generalization of Paillier's public-key system*, 2010.
