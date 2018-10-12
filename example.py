import pp_api
import os
from pp_vectorizer import pp_vectorizer as ppv
from pp_vectorizer.doc_organizer import TextFileIterator
from decouple import config

doc_folder = config("DOCS_PATH")
pp_pid = config("PP_PID")
server = config('PP_SERVER')
auth_data = (config("PP_USER"), config("PP_PASSWORD"))
language="de"

pp_client = pp_api.PoolParty(server=server, auth_data=auth_data, lang=language)

# This code shows how to:
# 1. extract the concepts present in a document
# 2. extract the positions in which these concepts are present
# 3. get which concepts are more broader (higher in the taxonomy) to each of them (i.e. the more general concepts)
for filename in os.listdir(doc_folder):
    filepath = os.path.join(doc_folder, filename)
    r = pp_client.extract_from_file(file=filepath,
                                    pid=pp_pid)

    # 1.  This gives back a list of dictionaries with data about the concepts present
    cpts = pp_client.get_cpts_from_response(r)

    # 3.  Here we construct a dictionary with the labels of the broaders, because URIs are not very nice for humans
    all_broaders = [b for c in cpts for b in c['transitiveBroaderConcepts']]
    label_dict_for_broaders = {x: pp_client.get_pref_labels([x], pid=pp_pid)[0]
                               for x in all_broaders}

    print("\nFile ", filename, "inlcudes the following concepts: ")
    for i,c in enumerate(cpts):
        print(i,":\t\"",c['prefLabel'], "\"(", c['uri'], ")",
              "\n\twhich is a particular of:", ", ".join([label_dict_for_broaders[x] for x in c['transitiveBroaderConcepts']]),
              "\n\ta total of", c['frequencyInDocument'], "times: ")
        # 2.
        if 'matchings' not in c.keys():
            continue
        for match in c['matchings']:
            print("\t\twith text: ",match['text']," in positions:",",".join([str(x[0]) for x in match['positions']]))



# This code creates a scikitleanr vectorizer for the documents in doc_folder and prints the vectors


vectorizer_parameters = {
    'pp':pp_client,
    'pp_pid':pp_pid,
    'ngram_range': (1, 3),
    'max_df': 0.51,
    'max_features': 1000,
    'use_terms': False,
    'related_prefix': None}


print('Preparing for extraction')
text_iter = TextFileIterator(doc_folder)
print('Text Iterator Prepared')
vectorizer = ppv.PPVectorizer(**vectorizer_parameters)
print('Vectorizer Prepared with params: {}'.format(vectorizer_parameters))
X = vectorizer.fit_transform(text_iter)
print("\n\nA total of ",
      len([x for x in vectorizer.vocabulary_ ]),
      "words are in the vocabulary")
print("The words are given by the URI of the concept."
      "If you inspect X.vocabulary_ you will find that some of them have 'broader ' as a prefix.",
      "These are words that aren't actually in the document, but which are broader concepts to concepts that are",
      "The prefix is added so that you know this distinction, and maybe treat them differently",
      "If you don't want to treat them differently, set 'broader_prefix':''  in the parameters",
      "If you set broader prefix to None, then broaders aren't considered at all in the extraction",
      sep="\n")

print("The vector representing the first document is:", str(X[0,:].todense()[0]))
