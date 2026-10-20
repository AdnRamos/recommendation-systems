# User-Based Nearest Neighbors

## Divisão do trabalho

**Pessoa 1 (implementada aqui):** matriz de avaliações, Pearson nos itens em comum,
seleção de até `k` vizinhos positivos, previsão por desvios em relação às médias,
Top-N e reprodução do exemplo do Slide 2.

**Pessoa 2 (próxima etapa):** conferir a base dummy em `data/`, criar uma divisão
reproduzível entre treino e teste, medir MAE/MSE/RMSE e NDCG@N, comparar com uma
referência simples e apresentar exemplos de acertos e erros.

## Executar o exemplo

Na pasta `implementations/userbased-nearestneighbor`:

```powershell
python example_slide.py
python -m unittest discover -s . -p "test_*.py"
```

## Interface para a próxima etapa

`load_ratings("data/ratings.csv")` devolve `dict[user_id][item_id] = nota`.
`UserBasedKNN(ratings, k=2, min_common=3)` recebe **somente notas de treino**.
`predict(user_id, item_id)` devolve `Prediction` ou `None` se não houver vizinhos
válidos. `recommend(user_id, n=5)` devolve previsões ordenadas para itens ainda
não vistos. Uma avaliação escondida para teste deve ser removida do dicionário
antes de criar o modelo; nunca pode participar da similaridade nem das médias.

Pearson usa as médias dos **itens em comum** do par de usuários. A previsão usa
a média de **todas as notas disponíveis no treino** de cada usuário. Só vizinhos
com similaridade positiva entram na previsão. A nota prevista é limitada à
escala de 1 a 5. Itens sem vizinhos ficam sem previsão e devem ser contabilizados
na cobertura da avaliação.

No exemplo do slide, as similaridades calculadas são `0,85` (User1), `0,71`
(User2), `0,00` (User3) e `-0,79` (User4). A página 9 parece trocar os valores
de User2 e User3. Com `k=2`, a previsão de Alice para Item5 é `4,872`.
