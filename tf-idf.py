import numpy as np
import string

class TFIDF:
    def __init__(self, documents, min_df = 0, max_df = None):
        if len(documents) == 0:
            raise ValueError("Il n'y a aucun document à analyser.")
        self.dict = {}
        self.IDF = {}
        self.documents = documents
        self.min_df = min_df
        self.max_df = max_df if max_df is not None else np.log(len(documents))
        self.build_dict()
    
    def tokenization(self,l : str) -> list[str]:
        txt = l.lower()
        translator = str.maketrans('', '', string.punctuation)
        txt = txt.translate(translator)
        lst = txt.split()
        return lst

    def calculate_tf(self, t : str) -> list[float]:
        tf = []
        nb_doc = len(self.documents)
        for i in range(0, nb_doc):
            nb_apparitions = 0
            lst = self.tokenization(self.documents[i])
            for mot in lst:
                if mot == t:
                    nb_apparitions = nb_apparitions + 1
            tf.append(nb_apparitions / len(lst))
        return tf
    
    def calculate_idf(self, t : str) -> float:
        nb_apparitions = 0
        for doc in self.documents:
            lst = self.tokenization(doc)
            if t in lst:
                nb_apparitions = nb_apparitions + 1
        self.IDF[t] = np.log(len(self.documents) / nb_apparitions)
        return np.log(len(self.documents) / nb_apparitions)
    
    def build_dict(self) -> dict[str,int]:
        index = 0
        for doc in self.documents:
            lst = self.tokenization(doc)
            for mot in lst:
                if mot not in self.dict:
                    mot_idf = self.calculate_idf(mot)  
                    if (mot_idf/np.log(len(self.documents)) < self.max_df) and (mot_idf/np.log(len(self.documents)) > self.min_df):
                        self.dict[mot] = index
                        index = index + 1
        return self.dict
    
    def tfidf_vectorizer(self):
        lignes = []
        tf_mot = {mot : self.calculate_tf(mot) for mot in self.dict}
        for i in range(0, len(self.documents)):
            arr = [0 for i in range(0, len(self.dict))]
            for mot in self.dict:
                arr[self.dict[mot]] = (tf_mot[mot])[i] * self.IDF[mot]
            lignes.append(arr)
        matrix = np.array(lignes)
        return matrix
    
    def get_dict(self):
        return self.dict
    
    def get_idf_normalise(self):
        #Plus le mot apparait, plus la valeur affichée sera faible (0 pour les mots qui apparaissent partout, 1 si le mot n'apparait qu'une fois)
        return {mot : valeur/np.log(len(self.documents)) for mot,valeur in self.IDF.items()}
                
