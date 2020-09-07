<h4>Treinamento de um classificador linear simples</h4>
<h2>O que é necessário</h3>
<ul>
  <li>Conjunto de dados (X, Y);</li>
  <li>Um modelo linear, que chamamos de M(x)</li>
  <li>Uma função de erro, que chamaremos de E(x,y)</li>
</ul>
<h2>O treinamento é sempre contabilizado em <strong>épocas</strong></h2>
<ul>
  <li>Uma época corresponde a utilização de todos os dados de treinamento por
    <b>apenas uma vez</b></li>
</ul>

<h2>Modelo linear bidimensional (2D)</h2>
<ul>
  <li>M(x1, x2) = tanh(ax1 + bx2 + c) -> Hiperplano pode ser extraído de ax1 + bx2 + c = 0</li>
  <li>Utilizada a função tangente hiperbólica porque funciona melhor para efeitos de treinamento do que a função sinal.</li>
</ul>

<h2>Função de erro</h2>
<ul>
  <li>E(x1,x2,y) = (M(x1,x2)-y)²</li>
</ul>

<h2>Dado que os parâmetros deste modelo são a, b e c, queremos então minimizar a função</h2>
<ul>
  <li>G(a,b,c) = (tanh(ax1+bx2+c)-y)²</li>
</ul>

<h2>Descida do gradiente</h2>
<ul>
  <li>[a,b,c] = [a,b,c] - lambda * deltaG(a,b,c)</li>
</ul>
