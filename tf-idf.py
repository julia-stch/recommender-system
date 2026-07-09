import numpy as np
import string

class TFIDF:
    def __init__(self, documents, min_df = 0, max_df = None):
        if len(documents) == 0:
            raise ValueError("Il n'y a aucun document à analyser.")
        self.dict = {}
        self.IDF = {}
        self.TF = {}
        self.documents = documents
        self.min_df = min_df
        self.max_df = max_df if max_df is not None else np.log(len(documents))
    
    def tokenization(self,l : str) -> list[str]:
        txt = l.lower()
        translator = str.maketrans('', '', string.punctuation)
        txt = txt.translate(translator)
        lst = txt.split()
        return lst

    def build_dict(self):
        dict_tf = {}
        dict_apparitions = {}
        for i in range(0, len(self.documents)):
            lst = self.tokenization(self.documents[i])
            ens = list(set(lst)) #Pour se débarasser des doublons
            for mot in lst:
                if mot in dict_apparitions:
                    dict_apparitions[mot][i] = dict_apparitions[mot][i] + 1
                else:
                    dict_apparitions[mot] = [0 for j in range(0, len(self.documents))]
                    dict_apparitions[mot][i] = 1

            for mot in ens:
                if mot in dict_tf:
                    dict_tf[mot] = dict_tf[mot] + 1
                else:
                    dict_tf[mot] = 1

        self.IDF = {mot : np.log(len(self.documents)/tf) for mot, tf in dict_tf.items()}
        self.TF = dict_apparitions
        index = 0
        for mot, idf in self.IDF.items():
            if (idf/np.log(len(self.documents)) < self.max_df) and (idf/np.log(len(self.documents)) > self.min_df):
                self.dict[mot] = index
                index = index + 1

    
    def tfidf_vectorizer(self):
        lignes = []
        for i in range(0, len(self.documents)):
            arr = [0 for i in range(0, len(self.dict))]
            for mot in self.dict:
                arr[self.dict[mot]] = (self.TF[mot])[i] * self.IDF[mot]
            lignes.append(arr)
        matrix = np.array(lignes)
        return matrix
    
    def get_dict(self):
        return self.dict
    
    def get_idf_normalise(self):
        #Plus le mot apparait, plus la valeur affichée sera faible (0 pour les mots qui apparaissent partout, 1 si le mot n'apparait qu'une fois)
        return {mot : valeur/np.log(len(self.documents)) for mot,valeur in self.IDF.items()}
    
    def cosine_similarity(self, matrix):
        normes = np.linalg.norm(matrix, axis=1, keepdims=True) 
        normes[normes == 0] = 1  
        matrix_normalisee = matrix / normes
        return matrix_normalisee @ matrix_normalisee.T 
    
    def find_similarities(self, nom_film : str, liste_films : str, matrix, n : int) -> list[str]:
        index = 0
        try:
            index = liste_films.index(nom_film)
        except ValueError:
            print("Il n'existe pas de film dont le nom correspond à celui fourni.")
        lst = list(enumerate(matrix[index]))
        lst = sorted(lst, key = lambda x: x[1], reverse= True)
        result = []
        for i in range(1,n + 1):
            result.append(liste_films[lst[i][0]])
        return result

                
