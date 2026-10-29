"""IDFace — identificação facial sobre templates cifrados.

Implementação da variante Paillier de:
  "IDFace: Face Template Protection for Efficient and Secure Identification"
  (arXiv:2507.12050, 2025).

Ideias centrais do artigo:
1. Transformação *quase isométrica* T_k: mantém as k componentes de maior
   magnitude do template (norma 1) e troca cada uma pelo seu sinal, gerando um
   vetor ternário em {-1, 0, 1}^d. O produto interno vira soma de inteiros.
2. Cada template ternário é separado em x = x⁺ − x⁻ (dois vetores binários) e
   ⟨x, y⟩ = ⟨x⁺,y⁺⟩ + ⟨x⁻,y⁻⟩ − ⟨x⁺,y⁻⟩ − ⟨x⁻,y⁺⟩.
3. Empacotamento: m templates por texto claro, como dígitos na base p > β:
   X_j = Σ_i p^i · x_{i,j}. Com cifragem aditivamente homomórfica, uma única
   multiplicação de ciphertexts soma as pontuações dos m templates de uma vez.
4. A consulta (em claro no servidor local) é ternária, então o produto interno
   cifrado usa só multiplicações de ciphertexts — nenhuma exponenciação.
5. Dois servidores: o *local* guarda a base cifrada e a chave pública; o *de
   chave* guarda só a chave secreta, decifra as pontuações, e responde apenas
   qual posição passou do limiar τ.

Extensões desta implementação (não estão no artigo):
* eliminação real de cadastros (LGPD art. 18) com *token de remoção* de uso
  único, registrado no servidor de chave;
* autenticação por token, limite de requisições e log de auditoria no
  servidor de chave, para que ele não vire um oráculo de decifração.
"""
