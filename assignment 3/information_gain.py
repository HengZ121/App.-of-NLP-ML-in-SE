import csv
import math
import statistics

'''
I am doing the challenge of using the fewest lines to implement functions in this code file
'''
__author__ = "Heng Zhang"
__email__ = "hzhan274@uOttawa.ca"

### Read data from csv
csv_content = []
with open ("result.csv", 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        csv_content.append(row)
print("Dataset Loaded.")

features = []
labels = []
for sentence in csv_content:
    features.append([sentence[x] for x in range(2,22)])
    labels.append(sentence[1])
print("Features and Labels Extracted")

# Calculate H(C) main entropy
# H(C) = -Σ(p*log2(p)) = -(p(NomAnaph)*log2(p(NomAnaph)) + p(ClauseAnaph)*log2(p(ClauseAnaph)))
number_of_nomanaph = len([x for x in labels if x == "NomAnaph"])
total_entropy = -1*(number_of_nomanaph/len(labels)*math.log2(number_of_nomanaph/len(labels)) + ((len(labels)-number_of_nomanaph)/len(labels)*math.log2(((len(labels)-number_of_nomanaph)/len(labels)))))
print("H(C) is computed as: %f" %(total_entropy))


'''
  Calculate info entropy without each feature:
  H(C|T) = Σ(split(T)*Σ(P(t)*log2(t))) for all possible t or = P(t)H(C|t) + P(t')H(C|t')
  Then calculate info gain using this equation: ヽ(✿ﾟ▽ﾟ)ノᵀʰᶦˢ ᵃˢˢᶦᵍⁿᵐᵉⁿᵗ ᶦˢ ᶠᶦⁿᵃˡˡʸ ᵍᵒᶦⁿᵍ ᵗᵒ ᵃⁿ ᵉⁿᵈ
                                                 (ー`n´ー) ᴰᵃᵐⁿ ᵗʰᵉʳᵉ ᶦˢ ᵃ ʳᵉᵖᵒʳᵗ ˡᵉᶠᵗ
  Info(T) = H(C) - H(C|T)
'''
res = dict()
for feature in range(0, 20):
    possible_values=[]
    sub_entropy = 0
    for instance in features:
        if not instance[feature] in possible_values:
            possible_values.append(instance[feature])
            labels_of_value = [labels[x] for x in range(0, len(features)) if features[x][feature] == instance[feature]]
            n_of_nomanaph = len([x for x in labels_of_value if x == "NomAnaph"])
            sub_entropy = sub_entropy + (len(labels_of_value)/len(labels)*(-1)*(n_of_nomanaph/len(labels_of_value)*math.log2(n_of_nomanaph/len(labels_of_value) if not n_of_nomanaph == 0 else 78047065) + ((len(labels_of_value)-n_of_nomanaph)/len(labels_of_value)*math.log2((len(labels_of_value)-n_of_nomanaph)/len(labels_of_value) if not (len(labels_of_value)-n_of_nomanaph) == 0 else 78047065))))
            ### H(C|T)  =     Σ (        P(t)                                 * H(C|T)                       )              Note: 78047065 means "N/A" in utf-8 encoding
    res[total_entropy - sub_entropy] = feature + 1 ### create a dictionary to store the result
for ig in sorted(res, reverse = True):
    print("Feature %d's information gain is %f" %(res[ig], ig))
