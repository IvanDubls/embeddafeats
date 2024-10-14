# embeddafeats

BertForFeatures is a Python class designed for generating text embeddings using a BERT model and applying Principal Component Analysis (PCA) for dimensionality reduction. This tool is ideal for transforming textual data into numerical features suitable for machine learning models.

## Requirements

- Python 3.x
- [sentence-transformers](https://www.sbert.net/)
- scikit-learn
- numpy
- pandas
- tqdm

## Installation

Install the required packages using the following command:

`pip install sentence-transformers scikit-learn numpy pandas tqdm`

## Usage Example

**Example Scenario:**

Suppose you have the following texts:

- "This is a sample sentence."
- "Here is another example."
- "BERT models are powerful for NLP tasks."

You can generate embeddings reduced to 5 dimensions using PCA and view the resulting DataFrame.

```python
from embeddafeats import BertForFeatures
import pandas as pd

# Initialize the BertForFeatures class
bert_features = BertForFeatures(model="sentence-transformers/all-MiniLM-L6-v2")

# Sample texts
texts = pd.Series([
    "That rug really tied the room together.",
    "that's a great plan walter, that's really ingenious if I understand it correctly.",
    "Sometimes you eat the bear, and sometimes, well, he eats you."
])

# Generate embeddings
embeddings_df = bert_features.get_embedding_features(
    texts=texts,
    batch_size=8,
    shrink_num=5,  # Reduce to 5 dimensions using PCA
    feature_prefix="feature_"
)

print(embeddings_df.head())
```

**Resulting DataFrame:**

|    | feature_1 | feature_2 | feature_3 | feature_4 | feature_5 |
|----|-----------|-----------|-----------|-----------|-----------|
| 0  | 0.123456  | 0.234567  | 0.345678  | 0.456789  | 0.567890  |
| 1  | 0.223344  | 0.334455  | 0.445566  | 0.556677  | 0.667788  |
| 2  | 0.323232  | 0.424242  | 0.525252  | 0.626262  | 0.727272  |

## License

This project is licensed under the MIT License.

---

# embeddafeats

BertForFeatures — это класс на Python, предназначенный для генерации эмбеддингов текста с использованием модели BERT и применения метода главных компонент (PCA) для снижения размерности. Этот инструмент идеально подходит для преобразования текстовых данных в числовые признаки для моделей машинного обучения.

## Требования

- Python 3.x
- [sentence-transformers](https://www.sbert.net/)
- scikit-learn
- numpy
- pandas
- tqdm

## Установка

Установите необходимые пакеты с помощью следующей команды:

`pip install sentence-transformers scikit-learn numpy pandas tqdm`

## Пример использования

**Юзкейс:**

Предположим, у вас есть следующие тексты:

- "Кто там первый, нельзя ли рыть тропу прямо, а?"
- "Хз, может там кто косоногий или косолапый, лол."
- "О Боже, я шерстяной волчара, как я хорош, как мощны мои лапищи."

Вы можете сгенерировать эмбеддинги, сниженные до 5 измерений с помощью PCA, и просмотреть полученный DataFrame.

```python
from embeddafeats import BertForFeatures
import pandas as pd

# Инициализация класса BertForFeatures
bert_features = BertForFeatures(model="sentence-transformers/all-MiniLM-L6-v2")

# Пример текста
texts = pd.Series([
"Кто там первый, нельзя ли рыть тропу прямо, а?",
"Хз, может там кто косоногий или косолапый, лол.",
"О Боже, я шерстяной волчара, как я хорош, как мощны мои лапищи."
])

# Генерация эмбеддингов
embeddings_df = bert_features.get_embedding_features(
    texts=texts,
    batch_size=8,
    shrink_num=5,  # Снизить до 5 измерений с помощью PCA
    feature_prefix="feature_"
)

print(embeddings_df.head())
```

**Полученный DataFrame:**

|    | feature_1 | feature_2 | feature_3 | feature_4 | feature_5 |
|----|-----------|-----------|-----------|-----------|-----------|
| 0  | 0.123456  | 0.234567  | 0.345678  | 0.456789  | 0.567890  |
| 1  | 0.223344  | 0.334455  | 0.445566  | 0.556677  | 0.667788  |
| 2  | 0.323232  | 0.424242  | 0.525252  | 0.626262  | 0.727272  |

## Лицензия

Этот проект лицензирован под лицензией MIT.
