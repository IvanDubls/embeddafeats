from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from typing import Sequence
import numpy as np
import pandas as pd
import tqdm.notebook as tq

class BertForFeatures(object):
    """
    A class for generating embeddings from text using a BERT model and applying PCA for dimensionality reduction.

    This class uses a `SentenceTransformer` model to encode text into embeddings, with optional dimensionality reduction
    using Principal Component Analysis (PCA). It is useful for extracting numerical features from text data for use in
    machine learning models.

    Attributes
    ----------
    model : SentenceTransformer
        The `SentenceTransformer` model used for generating embeddings.
    PCAer : PCA
        The PCA model used for dimensionality reduction (initialized when PCA is applied).

    Dependencies
    ------------
    - `SentenceTransformer` from `sentence_transformers`
    - `PCA` from `sklearn.decomposition`
    - `numpy` as `np`
    - `pandas` as `pd`
    - `tqdm` as `tq`
    """

    def __init__(self,
                 model: str = "deepvk/USER-bge-m3", 
                 **kwargs):
        """
        Initializes the `BertForFeatures` instance with a specified BERT model.

        Parameters
        ----------
        model : str, optional
            The pre-trained BERT model to use for generating embeddings.
            Default is `"deepvk/USER-bge-m3"`.
        **kwargs
            Additional keyword arguments to pass to the `SentenceTransformer` initializer.

        Returns
        -------
        None

        Notes
        -----
        The model is loaded using `SentenceTransformer`, which can load models from the Hugging Face model hub or local paths.
        """
        self.model = SentenceTransformer(model, **kwargs)
            
    def _chunks(self, 
                array: list,
                chunk_size: int):
        """
        Yields successive chunks of the specified size from the array.

        Parameters
        ----------
        array : list
            The list to be divided into chunks.
        chunk_size : int
            The size of each chunk.

        Yields
        ------
        list
            A chunk of the array with length up to `chunk_size`.

        Examples
        --------
        >>> list(self._chunks([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
        """
        for i in range(0, len(array), chunk_size):
            yield array[i:i + chunk_size]
    
    def _apply_pca(self,
                   data: np.array,
                   shrink_num: int):
        """
        Applies PCA to reduce the dimensionality of the data.

        Parameters
        ----------
        data : np.array
            The data to which PCA will be applied.
        shrink_num : int
            The number of principal components to keep.

        Returns
        -------
        np.array
            The data transformed by PCA with reduced dimensionality.

        Notes
        -----
        This method initializes a PCA model and fits it to the data, transforming it to have the specified number of components.
        The PCA model is stored in the attribute `PCAer`.
        """
        self.PCAer = PCA(n_components=shrink_num)
        return self.PCAer.fit_transform(data)
    
    def get_embedding_features(self,
                               texts: Sequence[str],
                               batch_size: int = 8, 
                               shrink_num: int = 10, 
                               feature_prefix: str = "embed_"):
        """
        Generates embedding features from a sequence of texts using the BERT model.

        This method processes the texts in batches, encodes them using the initialized `SentenceTransformer` model,
        optionally applies PCA for dimensionality reduction, and returns the embeddings as a pandas DataFrame.

        Parameters
        ----------
        texts : Sequence[str] or pandas.Series
            A sequence of text strings to generate embeddings for. Can be a list or pandas Series.
        batch_size : int, optional
            The number of texts to process in each batch. Default is `8`.
        shrink_num : int or None, optional
            The number of dimensions to reduce the embeddings to using PCA.
            If `None`, PCA is not applied and the original embedding dimensions are retained. Default is `10`.
        feature_prefix : str, optional
            The prefix to use for the feature column names in the output DataFrame. Default is `"embed_"`.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the embedding features, with one row per input text and columns named with the specified `feature_prefix`.
            If the input `texts` were a pandas Series, the index of the output DataFrame matches the index of the input Series.

        Notes
        -----
        - The embeddings are normalized to unit length (L2 norm) during encoding.
        - PCA is applied to reduce the dimensionality of the embeddings if `shrink_num` is specified.
        - Progress of the encoding is displayed using `tqdm` for progress bars.

        Examples
        --------
        >>> bert_features = BertForFeatures()
        >>> texts = ["This is a sentence.", "This is another sentence."]
        >>> embeddings_df = bert_features.get_embedding_features(texts)
        >>> embeddings_df.head()
           embed_1  embed_2  ...  embed_10
        0   0.1234  -0.5678  ...    0.4321
        1  -0.2345   0.6789  ...   -0.5432
        """
        if isinstance(texts, pd.Series):
            saved_indices = texts.index
            texts = texts.tolist()
        else:
            saved_indices = None
            
        batches = self._chunks(array=texts,
                               chunk_size=batch_size)
        embeddings = []
        for batch in tq.tqdm(batches):
            new_embeddings = np.array(self.model.encode(batch, 
                                           normalize_embeddings=True))
            embeddings += [new_embeddings]
        embeddings = np.concatenate(embeddings)

        if shrink_num is not None:
            embeddings = self._apply_pca(data=embeddings, 
                                         shrink_num=shrink_num)
        embeddings = pd.DataFrame(embeddings, 
                                  index=saved_indices,
                                  columns=[feature_prefix + str(i + 1) for i in range(len(embeddings[0]))])
        return embeddings
