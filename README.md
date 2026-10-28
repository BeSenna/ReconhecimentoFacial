# Reconhecimento facial em tempo real com uso de webcam
<br>
<br>

**1. Instalando pacotes**

1. Utilize o comando "python -m pip install -r requirements.txt". 
2. De um "pip list" para ver se tudo foi baixado, do contrário, terá de baixar pip a pip o que falta.
3. O "Mingw" é uma exceção, pois deve ser instalado pelo chocolatey. Mais informações abaixo.

<br>
<br>
<br>

**2. Mingw**

1. Instale o chocolatey.
2. Dê "choco install mingw" no prompt.
<br>

**Caso necessário colocar o mingw no PATH(Windows64bits), faça os seguintes passos:**

1. Vá no C:\ do seu computador.
2. Clique em visualizar.
3. Clique em mostrar.
4. Clique em "Items Ocultos".
5. Isso fará com que sua pasta "ProgramData" seja visivel, entre nessa pasta.
6. Procure a pasta mingw64.
7. Entre nessa pasta e procure nela a pasta "bin".
8. Entre na pasta bin, copie o caminho e coloque ele no PATH, seguindo as mesmas instruções que o da configuração de path do "cmake".

<br>
<br>
<br>

**3. Cmake**

1. pip install cmake (o cmake precisa ser baixado e configurado no path do sistema, apenas dar "pip install cmake" não funciona)
2. Baixe: https://cmake.org/download/
3. Durante a instalação, selecione a opção "Add CMake to the system PATH for all users".
<br>

**Ou**

Após baixar o cmake, coloque a pasta em algum diretório após desempacotar, então copie o caminho e cole no PATH em variáveis de ambiente, seguindo a ordem abaixo:

1. Variáveis de ambiente.
2. Clicar 2x em PATH.
3. Clicar em "novo".
4. Colar o caminho.
5. Dar "ok" em todas as opções até todas as janelas fecharem.

<br>
<br>
<br>

**4. Instalando o Dlib**

Caso usar pip install dlib não seja o suficiente, você pode tentar o seguinte passo a passo:

1. Verifique a sua versão do Python e a arquitetura do seu sistema operacional.
2. Sabendo as duas informações mencionadas acima, procure o arquivo dlib correspondente a sua versão do python e arquitetura do seu sistema no repositório Dlib do GitHub, ou na Hugging Face, ou apenas dê as informações para o ChatGPT que ele certamente te passará o link correto. Por exemplo: link para baixar arquivo dlib para python versão 3.12.6 com sistema Windows 64 bits.
3. Após baixar o arquivo, coloque-o em uma pasta que contenha apenas ele e copie o caminho do arquivo.
4. Navegue pelo prompt(O do VSCode mesmo) até a pasta em que se encontra o arquivo, digitando "cd caminho-do-arquivo".
5. Faça um pip install com o nome completo do arquivo, por exemplo "pip install dlib-19.24.99-cp312-cp312-win_amd64.whl".
6. Então dê "pip install face-recognition".
7. Volte para a pasta do sistema que está montando.

**Obs:** Caso seja necessário fazer downgrade na versão do numpy(pip uninstall -y numpy), fazer dentro da pasta dlib. Geralmente, o dlib funciona no numpy versão "pip install numpy==1.26.4" ou "pip install numpy==1.23.5".
<br>
<br>

**Caso ainda assim o face_recognizer não esteja sendo reconhecido, utilize os comandos abaixo:**

1. pip uninstall face_recognition_models -y
2. pip install git+https://github.com/ageitgey/face_recognition_models

<br>
<br>
<br>

**5. Como usar**

1. Tire uma ou mais fotos de um ou mais rostos e salve no formato jpg(preferencialmente).
2. Crie uma pasta "images" e coloque a(s) foto(s) nessa pasta.
3. Coloque o nome de cada pessoa em sua respectiva foto.
4. Rode a aplicação detector.py(python detector.py) para treinar o programa. Ela irá detectar os rostos nas fotos postas na pasta "images".
5. Após detectados os rostos, rode a aplicação reconhecer.py(python reconhecer.py). Ela irá ligar automaticamente a webcam e identificará os rostos que aparecem nela, colocando o nome respectivo a cada rosto. Caso seja um rosto desconhecido, ou seja, que não passou pelo treinamento, o programa colocará "desconhecido" abaixo do rosto.

<br>
<br>
<br>

**6. Linux (Ubuntu 26.04)**

No Linux não é preciso Chocolatey, MinGW nem configurar PATH: o compilador e o CMake vêm do apt.

1. Instale as dependências do sistema:
   `sudo apt update && sudo apt install -y build-essential cmake python3-venv python3-dev libopenblas-dev liblapack-dev git`
2. Crie e ative um ambiente virtual (o Ubuntu bloqueia `pip install` no Python do sistema):
   `python3 -m venv .venv && source .venv/bin/activate`
3. Instale os pacotes (o dlib é compilado na hora, pode levar alguns minutos):
   `python -m pip install -r FaceRecognition/face_recognizer/requirements.txt`
4. Garanta acesso à webcam: `sudo usermod -aG video $USER` (depois faça logout/login) e confira se existe `/dev/video0`.
5. Rode os scripts de dentro da pasta `FaceRecognition/face_recognizer`, como descrito no item 5.
