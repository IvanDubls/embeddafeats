# embeddafeats

BertForFeatures is a Python class designed for generating text embeddings using a BERT model and applying Principal Component Analysis (PCA) for dimensionality reduction. This tool is ideal for transforming textual data into numerical features suitable for machine learning models.

## Features

- Generate embeddings from text using a pre-trained BERT model.
- Process large datasets efficiently with batch processing.
- Optional PCA for dimensionality reduction of embeddings.
- Output embeddings as a pandas DataFrame for easy integration with data workflows.

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

1. **Import the necessary modules:**

   Import `BertForFeatures` and `pandas` into your Python script.

2. **Initialize the class:**

   Create an instance of `BertForFeatures`, specifying the desired BERT model (e.g., `"sentence-transformers/all-MiniLM-L6-v2"`).

3. **Prepare your text data:**

   Organize your texts into a pandas Series or a list of strings.

4. **Generate embeddings:**

   Use the `get_embedding_features` method to generate embeddings from your texts. You can specify parameters like `batch_size`, `shrink_num` for PCA dimensionality reduction, and `feature_prefix` for naming the output features.

5. **View the results:**

   The method returns a pandas DataFrame containing the embeddings, which you can analyze or integrate into your machine learning models.

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

## Возможности

- Генерация эмбеддингов из текста с помощью предобученной модели BERT.
- Эффективная обработка больших данных с пакетной обработкой.
- Опциональное применение PCA для снижения размерности эмбеддингов.
- Вывод эмбеддингов в формате pandas DataFrame для удобной интеграции с рабочими процессами данных.

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

1. **Импортируйте необходимые модули:**

   Добавьте в ваш Python-скрипт импорт `embeddafeats` и `pandas`.

2. **Инициализируйте класс:**

   Создайте экземпляр `BertForFeatures`, указав желаемую модель BERT (например, `"sentence-transformers/all-MiniLM-L6-v2"`).

3. **Подготовьте текстовые данные:**

   Организуйте ваши тексты в pandas Series или список строк.

4. **Сгенерируйте эмбеддинги:**

   Используйте метод `get_embedding_features` для генерации эмбеддингов из ваших текстов. Вы можете указать такие параметры, как `batch_size`, `shrink_num` для снижения размерности с помощью PCA и `feature_prefix` для названия выходных признаков.

5. **Просмотрите результаты:**

   Метод вернет pandas DataFrame с эмбеддингами, которые вы можете анализировать или интегрировать в ваши модели машинного обучения.

**Пример ситуации:**

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
