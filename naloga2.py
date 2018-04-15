from collections import Counter
from itertools import combinations
from random import randint
from unidecode import unidecode
import math
import csv

import matplotlib.pyplot as plt
import numpy as np


class KMedoidsClustering:

    def __init__(self, file_names, folder='data'):

        self.languages = file_names
        self.data = {}
        self.all = {}
        self.cosine_similarities = {}

        # read .csv file
        with open(folder + "/" + file_names) as csvfile:
            readCSV = csv.reader(csvfile, delimiter="\t")
            next(readCSV)
            i = 0
            for row in readCSV:
                #print(row[2])
                #next(readCSV)

                if (len(row) > 3) :
                    text = self.preprocess_text2(row[2])
                    #print(str(i) + " new data\n\n")

                    # kmers
                    self.data[i] = Counter(self.kmers(text, 3))
                    self.all[i] = row

                    # for every text compute kmers (relative frequency)
                    self.get_frequency(i)

                    i += 1

        print(len(self.data.keys()))

        # for all combinations calculate similarities (so we don't have to do it again)
        #for l1, l2 in combinations(self.languages, r=2):
        #    self.cosinus_similarity(l1, l2)



    def preprocess_text(self, file, folder):
        # preprocess text (all to lowercase, remove commas, dots, ..)

        f = open(folder + "/" + file + ".txt", "rt", encoding="utf8").read()

        # remove commas and dots, numbers; convert to lowercase
        text = ''.join(filter(lambda s: s.isalpha() or s.isspace(), f)).lower()

        # remove newlines and empty lines, but keep spaces
        text = ' '.join([line for line in text.split('\n') if line.strip() != ''])

        return unidecode(text)

    def preprocess_text2(self, text_original):
        # preprocess text (all to lowercase, remove commas, dots, ..)

        # remove commas and dots, numbers; convert to lowercase
        text = ''.join(filter(lambda s: s.isalpha() or s.isspace(), text_original)).lower()

        # remove newlines and empty lines, but keep spaces
        text = ' '.join([line for line in text.split('\n') if line.strip() != ''])

        return unidecode(text)



    def kmers(self, text, k=3):
        # split text to k-letter groups

        for i in range(len(text) - k + 1):
            yield text[i:i + k]


    def get_frequency(self, key):
        # compute relative freq for every 'kmer'

        elements_num = sum(Counter(self.data[key]).values())

        for terka in self.data[key].keys():
            self.data[key][terka] = self.data[key][terka] / elements_num


    def cosinus_similarity(self, text1, text2):
        # get cosine similarity  between two texts
        #print(text1, text2)

        product_sum = 0
        text1_length_sum = 0
        text2_length_sum = 0

        for key in self.data[text1]:
            #print(key)
            if key in self.data[text2]:
                # if some 'terka' is present in both texts, add to sum (otherwise we have x * 0 = 0)
                product_sum += self.data[text1][key] * self.data[text2][key]

            # while we iterate through all elements in text1, also get sum of text1_length
            text1_length_sum += self.data[text1][key] ** 2


        # get sum of text2_length for second text
        for key in self.data[text2]:
            text2_length_sum += self.data[text2][key] ** 2

        #print(text1_length_sum, text2_length_sum)
        sum_ik = math.sqrt(text1_length_sum) * math.sqrt(text2_length_sum)

        if sum_ik > 0:
            cosine_sim = product_sum / (sum_ik)
        else:
            return 0

        # save value so we don't have to do it again
        #self.cosine_similarities[text1 + '-' + text2] = cosine_sim

        return cosine_sim


    def k_medoids(self, k=5):
        # k-medoids clustering

        self.medoids = {}

        # randomly select k-starting self.medoids
        for i in range(0, k):
            while True:
                random_lang = self.languages[randint(0, len(self.languages)-1)]

                if not random_lang in self.medoids:
                    self.medoids[random_lang] = []
                    break

        # while medoids change, keep going!
        medoids_change = True
        while medoids_change:

            # for every instance that is not medoid, find closest medoid
            self.find_closest_medoid()
            #print('CLUSTERS', self.medoids)

            # find best medoid for every cluster
            medoids_change = self.find_center_medoid()

        # compute silhouette
        return self.silhouette_all()


    def copy_items(self, items):
        # helper method for copying dict array to new array (so that we don't create just a new pointer)

        new_items = []
        for item in items:
            new_items.append(item)

        return new_items


    def silhouette_all(self):
        # get average silhouttte from every point (all clusters)

        silhouette_sum = 0
        no_silhoutte = 0

        # for every element in every cluster
        for key in self.medoids:
            # all elements in cluster (including key)
            cluster = self.copy_items(self.medoids[key])
            cluster.append(key)

            # if cluster only has 1 element, silhouette = 0, so we only increase counter
            if len(cluster) == 1:
                no_silhoutte += 1

            else:
                for item in cluster:
                    # get silhoutte for every item in cluster
                    # convert similarity to distance
                    ai = 1 - self.avg_cluster_similarity_to_lang(item, cluster)
                    bi = 1 - self.avg_nearest_cluster_similarity(item, key)

                    # silhouette for this one item
                    si = (bi - ai) / max(ai, bi)
                    #print('Silhouette:', si, bi, ai)
                    silhouette_sum += si
                    no_silhoutte += 1

        # return avg. silhouette
        return silhouette_sum / no_silhoutte


    def avg_nearest_cluster_similarity(self, item, own_cluster_key):
        # get average similarity to the most similar cluster

        max_similarity = None

        for key in self.medoids:

            # only compute distance to other clusters (no item's cluster)
            if key != own_cluster_key:
                cluster = self.copy_items(self.medoids[key])
                cluster.append(key)

                # average distance of 'element' to elemnts from other cluster
                dist = self.avg_cluster_similarity_to_lang(item, cluster)

                if max_similarity is None or max_similarity < dist:
                    max_similarity = dist

        return max_similarity


    def find_closest_medoid(self):
        # find closest medoid for every language and add languages to self.medoids' array

        # first empty every cluster
        for key in self.medoids.keys():
            self.medoids[key] = []

        for lang in self.languages:
            if lang not in self.medoids:
                max_similarity = None
                max_medoid = None

                # compare similarity to all self.medoids and find the closest one (use already computed similarities between langs)
                for key in self.medoids.keys():
                    combo_key = self.find_key_combo(lang, key)

                    if combo_key in self.cosine_similarities:

                        if max_similarity is None or self.cosine_similarities[combo_key] > max_similarity:
                            max_similarity = self.cosine_similarities[combo_key]
                            max_medoid = key

                self.medoids[max_medoid].append(lang)


    def find_center_medoid(self):
        # for every cluster find the best medoid (biggest avg similarity to all other members of cluster)
        # return 'True' if medoids change, else return false

        new_medoids = {}
        medoids_change = False

        for key in self.medoids:

            # add medoid to other cluster members (so we have all cluster members in array)
            cluster_members = self.copy_items(self.medoids[key])
            cluster_members.append(key)
            max_avg_similarity = None
            max_lang = None

            if len(cluster_members) == 1:
                # if we only have one element in cluster (medoid) we can skip calculations
                new_medoids[cluster_members[0]] = []

            else:
                # more than 2 elements, compare avg_distances and pick minimum
                for lang in cluster_members:
                    dist = self.avg_cluster_similarity_to_lang(lang, cluster_members)

                    if max_avg_similarity is None or max_avg_similarity < dist:
                        max_avg_similarity = dist
                        max_lang = lang

                    elif max_avg_similarity == dist:
                        # if distance is the same for two elements, we have to flip a coin to choose one
                        rand_winner2 = randint(0, 1)

                        if rand_winner2 == 1:
                            max_lang = lang

                # add new medoid and cluster members (from cluster members remove new medoid)
                new_medoids[max_lang] = cluster_members
                new_medoids[max_lang].remove(max_lang)

        # check if new medoids are different from previous round
        if not self.medoids.keys() == new_medoids.keys():
            medoids_change = True

        # replace old medoids dict with new one
        self.medoids = new_medoids
        #print('NEW MEDOIDS', self.medoids)

        return medoids_change


    def avg_cluster_similarity_to_lang(self, lang, cluster):
        # average similarity for 'lang' element to all other elements in cluster
        #print('Avg distance, element:', lang, 'cluster:', cluster)
        distance_sum = 0
        no_of_elements = 0
        for item in cluster:
            key = self.find_key_combo(lang, item)

            if key in self.cosine_similarities:
                distance_sum += self.cosine_similarities[key]
                no_of_elements += 1

        return distance_sum/no_of_elements


    def find_key_combo(self, lang1, lang2):
        # since key in self.cosine_similarities is built by two lang names connected with '-',
        # sequence is unknown, so return the right one ('slo-eng' or 'eng-slo')

        if lang1 + '-' + lang2 in self.cosine_similarities:
            return lang1 + '-' + lang2
        else:
            return lang2 + '-' + lang1


    def draw_hist(self):

        data = self.all_silhouetts

        bins = np.linspace(math.ceil(min(data)),
                           math.floor(max(data)),
                           100)

        plt.hist(data)
        plt.show()


    def run(self):

        min_s = None
        min_clusters = None
        max_s = None
        max_clusters = None
        self.all_silhouetts = []

        # repeat 100x
        for i in range(0, 100):

            # do k-medoids
            avg_silhouette = self.k_medoids()
            self.all_silhouetts.append(avg_silhouette)

            if min_s is None or min_s > avg_silhouette:
                min_s = avg_silhouette
                min_clusters = self.medoids

            if max_s is None or max_s < avg_silhouette:
                max_s = avg_silhouette
                max_clusters = self.medoids

        print('\nMin silhouette:', min_s)
        print(min_clusters)

        print('\nMax silhouette:', max_s)
        print(max_clusters)

        self.draw_hist()

        return True


    def find_text_language(self, file, folder='data'):

        # text preparation (same as before)
        text = self.preprocess_text(file, folder)
        self.data[file] = Counter(self.kmers(text, 3))
        self.get_frequency(file)

        similarities = []
        similarities_langs = []

        # for every language get similarity with our text
        for lang in self.languages:
            similarities.append(self.cosinus_similarity(file, lang))
            similarities_langs.append(lang)

        # sum of all similarities
        sum_sim = sum(similarities)

        # best 3 similarities normalized with sum of all similarities
        best = sorted(zip([i/sum_sim for i in similarities], similarities_langs), reverse=True)[:3]
        print('\nBest 3 matches for file ' + file + '.txt:', best)


    def find_text_similar(self, file, folder='data'):

        # text preparation (same as before)
        #text = self.preprocess_text2(original_text)

        # text preparation (same as before)
        text = self.preprocess_text(file, folder)

        new_index = len(self.data.keys())
        self.data[new_index] = Counter(self.kmers(text, 3))
        self.get_frequency(new_index)

        print(len(self.data.keys()))

        similarities = []
        similarities_langs = []

        # for every language get similarity with our text
        for i in range(0, len(self.data.keys()) - 1):
            similarities.append(self.cosinus_similarity(new_index, i))
            similarities_langs.append(i)

        # sum of all similarities
        sum_sim = sum(similarities)

        avg_sim = sum_sim / len(similarities)

        # best 3 similarities normalized with sum of all similarities
        best = sorted(zip([i/sum_sim for i in similarities], similarities_langs), reverse=True)[:3]
        print('\nBest 3 matches for file ' + file + '.txt:', best)

        return avg_sim, best



# run

def returnBestText():
    kc1 = KMedoidsClustering("divorce_dn.csv", "borns")
    avg_1, best_1 = kc1.find_text_similar('divorce_example', 'borns')

    kc2 = KMedoidsClustering("bullying_dn.csv", "borns")
    avg_2, best_2 = kc2.find_text_similar('divorce_example', 'borns')

    kc3 = KMedoidsClustering("love_dn.csv", "borns")
    avg_3, best_3 = kc3.find_text_similar('divorce_example', 'borns')

    avg = [avg_1, avg_2, avg_3]
    best = [best_1, best_2, best_3]
    kc = [kc1, kc2, kc3]

    select = avg.index(max(avg))
    score, index = best[select][0]

    return kc[select].all[index]


#print(returnBestText())
