
import numpy as np

class Word2Vec :
    def __init__ ( self , model_file_path ) :
        self.__model_path__ = model_file_path
        self.__word_embeddings__ = None
        self.__word_to_id_map__ = None
        self.__id_to_word_map__ = None
        import pickle
        with open(model_file_path, 'rb') as model_file :
            word2vec_model = pickle.load(model_file)
            self.__word_embeddings__ = word2vec_model['WORD_EMBEDDINGS']
            self.__word_to_id_map__ = word2vec_model['WORD_TO_ID']
            self.__id_to_word_map__ = word2vec_model['ID_TO_WORD']
    
    def __getitem__ ( self , query ) :
        if ( isinstance(query, str) ) :
            query = query.lower()
            if ( not query in self.__word_to_id_map__ ) :
                return None
            return self.__word_embeddings__[self.__word_to_id_map__[query]]
        return None
    
    def CosineSimilarity ( self , word1 , word2 ) :
        vec1 , vec2 = None , None
        if ( isinstance(word1, str) ) : vec1 = self[word1]
        elif ( isinstance(word1, np.ndarray) ) : vec1 = word1
        else : return None
        if ( isinstance(word2, str) ) : vec2 = self[word2]
        elif ( isinstance(word2, np.ndarray) ) : vec2 = word2
        else : return None
        dot_product = abs(vec1.dot(vec2))
        mag1 = np.linalg.norm(vec1)
        mag2 = np.linalg.norm(vec2)
        similarity = dot_product / ( mag1 * mag2 )
        return round(similarity, 4)

    def GetMostSimilar ( self , query , limit = 10 ) :
        if ( isinstance(query, str) ) :
            query = query.lower()
            if ( not query in self.__word_to_id_map__ ) : return []
        similarities = list()
        for word in self.__word_to_id_map__ :
            similarities.append((word, self.CosineSimilarity(word, query)))
        similarities.sort(key = lambda x : -1*x[1])
        return similarities[:limit]
    
