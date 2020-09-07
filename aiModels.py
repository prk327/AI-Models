class PreProcessing:
    ''' The preprocessing class for word cleaning'''
    import re as _re
    _regex = _re.compile('[@_!#$%^&*()<>?/\|}{~:`]')
    def __init__(self):
        pass
    
    def ClearWords(self, string):
        '''make all the word lower case and clear all the special characters'''
        r = self._regex.sub(' ', string)
        return r.lower().strip()
        
    
class AlgoWorld:
    '''The various ML algo implementation from scratch'''
    import pandas as _pd
    import numpy as _np
    def __init__(self):
        pass

    def CalPro(self, DF, Y):
        '''This will calculate the conditional probability of Dependent variable'''
        # remove the duplicate rows based on all columns
        DF = DF.drop_duplicates()
        # reset the index
        DF = DF.reset_index(drop=True)
        c = list(DF.columns)
        c.remove(Y)
        pb = {}
        for items in DF[Y].unique():
            #This will calculate the normal probality P(items)
            py = DF[Y][DF[Y] == items].count() / DF[Y].count()
            pb[items] = py
            for col in c:
                #This will calculate the Condition probality P(levels|items)
                for levels in DF[col].unique():
                    cpx = DF[Y][(DF[Y] == items) & (DF[col] == levels)].count() / DF[Y][DF[Y] == items].count()
                    pb["{}|{}".format(levels,items)] = cpx
        return pb

    def NaiveBayes(self, DF, Y):
        '''This function will gives us the probability based on Naive Bayes theorem P(A|B) = P(B|A)*P(A)'''
        # remove the duplicate rows based on all columns
        DF = DF.drop_duplicates()
        # reset the index
        DF = DF.reset_index(drop=True)
        #This will give us the condition probability
        pb = self.CalPro(DF, Y)
        IV = DF.drop(columns=Y)
        nbdf = self._pd.DataFrame()
        #This will create a new df with condition probability
        CPDict = {}
        for items in DF[Y].unique():
            # This will fill all the element with its conditional probability
            CPDict[items] = IV.applymap(lambda x: pb["{}|{}".format(x,items)])
            #This will initialize a new df with 1 value
            d = self._pd.DataFrame(self._np.ones(IV.shape[0]), columns=['a'])
            vd = d["a"]
            #now we will multiply all the column and P(Y) as per asumption that all element are independent of each other
            for col in CPDict[items].columns:
                vd*=CPDict[items][col]
            #now we will multiply with P(Y) and add the value to the DF
            nbdf["P({}|{}...{})".format(items,CPDict[items].columns[0], CPDict[items].columns[-1])] = vd * pb[items]
        #now we will check the predicted probability and if it is greater than 50% we will assign that label
        for i in range(len(DF[Y].unique())- 1):
            nbdf["Prediction"]\
                 = nbdf.apply(lambda x: DF[Y].unique()[i]\
                      if x["P({}|{}...{})".format(DF[Y].unique()[i],CPDict[DF[Y].unique()[i]].columns[0], CPDict[DF[Y].unique()[i]].columns[-1])]\
                           > x["P({}|{}...{})".format(DF[Y].unique()[i+1],CPDict[DF[Y].unique()[i+1]].columns[0], CPDict[DF[Y].unique()[i+1]].columns[-1])]\
                                else DF[Y].unique()[i+1], axis=1)
        #now lets save the result in the original DF
        ODF = self._pd.concat([DF,nbdf], axis=1)
        #now for the sake of validation lets print CM
        print(self._pd.crosstab(ODF.loc[:,Y], ODF.loc[:,"Prediction"]))
        return ODF