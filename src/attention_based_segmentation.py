import nltk
from sklearn.cluster import KMeans
import numpy as np

from src.attention_utils import aggregate_attention


class Segmentor:

    def __init__(self, controller, prompts, num_segments, background_segment_threshold, res=32, background_nouns=[]):
        self.controller = controller
        self.prompts = prompts
        self.num_segments = num_segments
        self.background_segment_threshold = background_segment_threshold
        self.resolution = res
        self.background_nouns = background_nouns

        self.self_attention = aggregate_attention(controller, res=32, from_where=("up", "down"), prompts=prompts,
                                             is_cross=False, select=len(prompts) - 1)
        self.cross_attention = aggregate_attention(controller, res=16, from_where=("up", "down"), prompts=prompts,
                                              is_cross=True, select=len(prompts) - 1)
        tokenized_prompt = nltk.word_tokenize(prompts[-1])
        self.nouns = [(i, word) for (i, (word, pos)) in enumerate(nltk.pos_tag(tokenized_prompt)) if pos[:2] == 'NN']

    def __call__(self, *args, **kwargs):
        clusters = self.cluster()
        cluster2noun = self.cluster2noun(clusters)
        return cluster2noun

    def cluster(self):
        np.random.seed(1)
        resolution = self.self_attention.shape[0]
        attn = self.self_attention.cpu().numpy().reshape(resolution ** 2, resolution ** 2)
        kmeans = KMeans(n_clusters=self.num_segments, n_init=10).fit(attn)
        clusters = kmeans.labels_
        clusters = clusters.reshape(resolution, resolution)
        return clusters

    def cluster2noun(self, clusters):
        result = {}
        nouns_indices = [index for (index, word) in self.nouns]
        nouns_maps = self.cross_attention.cpu().numpy()[:, :, [i + 1 for i in nouns_indices]]
        normalized_nouns_maps = np.zeros_like(nouns_maps).repeat(2, axis=0).repeat(2, axis=1)
        for i in range(nouns_maps.shape[-1]):
            curr_noun_map = nouns_maps[:, :, i].repeat(2, axis=0).repeat(2, axis=1)
            normalized_nouns_maps[:, :, i] = (curr_noun_map - np.abs(curr_noun_map.min())) / curr_noun_map.max()
        for c in range(self.num_segments):
            cluster_mask = np.zeros_like(clusters)
            cluster_mask[clusters == c] = 1
            score_maps = [cluster_mask * normalized_nouns_maps[:, :, i] for i in range(len(nouns_indices))]
            scores = [score_map.sum() / cluster_mask.sum() for score_map in score_maps]
            result[c] = self.nouns[np.argmax(np.array(scores))] if max(scores) > self.background_segment_threshold else "BG"
        return result

    def get_background_mask(self, obj_token_index):
        clusters = self.cluster()
        cluster2noun = self.cluster2noun(clusters)
        mask = clusters.copy()
        obj_segments = [c for c in cluster2noun if cluster2noun[c][0] == obj_token_index - 1]
        background_segments = [c for c in cluster2noun if cluster2noun[c] == "BG" or cluster2noun[c][1] in self.background_nouns]
        for c in range(self.num_segments):
            if c in background_segments and c not in obj_segments:
                mask[clusters == c] = 0
            else:
                mask[clusters == c] = 1
        return mask

