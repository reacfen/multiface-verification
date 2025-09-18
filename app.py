# OpenCV library for computer vision capabilities
import cv2
# NumPy library for numerical computation
import numpy as np
# ONNXRuntime library for inference
import onnxruntime as ort

# Numba library for checking CUDA/NVIDIA GPU support
from numba import cuda
# InsightFace library for facial detection and analysis
from insightface.app import FaceAnalysis
# scikit-learn library
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

class DetectionModel:
    # Creates a 'DetectionModel' instance
    def __init__(self, model_name, model_root, det_size=(640, 640)):
        # Initialize app (it will load defaults first)
        if cuda.is_available() and 'CUDAExecutionProvider' in ort.get_available_providers():
            print('Running on the GPU...')
            self.app = FaceAnalysis(name=model_name, root=model_root, providers=['CUDAExecutionProvider'])
            self.ctx_id = 0
        else:
            print('Running on the CPU...')
            self.app = FaceAnalysis(name=model_name, root=model_root, providers=['CPUExecutionProvider'])
            self.ctx_id = -1
        # Prepare the app for usage
        self.app.prepare(ctx_id=self.ctx_id, det_size=det_size)

    # Generates a list of faces detected in a group image
    def get_group_faces(self, img_path):
        # Load image
        img = cv2.imread(img_path)
        # Raise an error on unsuccessful load of the image
        if img is None:
            raise ValueError(f'Image not found: {img_path}')
        # Detect faces and get embeddings
        faces = self.app.get(img)
        # Return the fetched faces from the image
        return faces

    # Returns the largest face (if any) detected in an image
    def get_largest_face(self, img_path):
        # Find faces in the specified image
        faces = self.get_group_faces(img_path)
        # If no faces were found, return None
        if len(faces) == 0:
            return None
        # Return the largest detected face in the image
        return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

class ProfileIdentifier:
    # Creates a 'ProfileIdentifier' instance
    def __init__(self, model):
        # Detection model
        self.model = model
        # Stores individual profiles
        self.profiles = []

    # Creates a profile/candidate to match against
    def create_profile(self, info, profile_img_paths=None, profile_embeddings=None):
        # Stores all embeddings attached to this profile
        embs = []
        # Add embeddings from image paths if available
        if profile_img_paths is not None:
            for path in profile_img_paths:
                # Gets the largest face detected in the image
                face = self.model.get_largest_face(path)
                # If no faces were found, raise an error
                if face is None:
                    raise ValueError(f'No face detected in {path}')
                # Otherwise, append the new face's embedding to the profile
                embs.append(face.normed_embedding)
        # Add embeddings explicitly if available
        if profile_embeddings is not None:
            embs.extend(profile_embeddings)
        # Picks the embedding closest to the rest
        def medoid_embedding(embs):
            D = cosine_distances(embs)
            medoid_idx = np.argmin(D.sum(axis=0))
            return embs[medoid_idx]
        # Create a new profile for the person with their newly generated facial and embedding data
        self.profiles.append({
            # Compute the overall mean embedding
            'embedding': medoid_embedding(embs),
            'info': info
        })

    # Finds profile matches within a specified group image using a given cosine threshold and appends them to a dictionary
    def find_matches(self, img_path, threshold=0.4, unique_per_profile=True):
        # Generate faces from the group image
        group_faces = self.model.get_group_faces(img_path)
        matched, unmatched, untagged = {}, {}, []
        # If no faces were found, return an empty list
        if not group_faces:
            return matched, unmatched, untagged
        # Detect and store any existing profile matches in the image
        for face in group_faces:
            face_emb = np.ravel(face.normed_embedding)  # Shape: (512,)
            max_sim_idx, max_sim = None, -1
            for idx, profile in enumerate(self.profiles):
                # Compute the cosine similarity
                sim = np.dot(profile['embedding'], face_emb)
                # Record the profile with the highest similarity
                if max_sim_idx is None or sim > max_sim:
                    max_sim_idx, max_sim = idx, sim
            # Round the fields of the bounding box to the nearest integer
            bbox = np.round(face.bbox).astype(np.int32)
            # Compare the profile with the highest similarity against a threshold and classify it as matched or unmatched
            if max_sim_idx is not None and max_sim > threshold:
                if unique_per_profile:
                    # Keep only the *best* match for each profile
                    if max_sim_idx not in matched or max_sim > max(matched[max_sim_idx]['sims']):
                        if max_sim_idx in matched:
                            untagged.extend(matched[max_sim_idx]['bboxes'])
                        matched[max_sim_idx] = {
                            'bboxes': [bbox],
                            'sims': [max_sim],
                            'info': self.profiles[max_sim_idx]['info']
                        }
                    else:
                        untagged.append(bbox)
                else:
                    # Allow multiple matches
                    matched.setdefault(max_sim_idx, {
                        'bboxes': [],
                        'sims': [],
                        'info': self.profiles[max_sim_idx]['info']
                    })
                    matched[max_sim_idx]['bboxes'].append(bbox)
                    matched[max_sim_idx]['sims'].append(max_sim)
            else:
                unmatched.setdefault(max_sim_idx, {
                    'bboxes': [],
                    'info': self.profiles[max_sim_idx]['info']
                })
                unmatched[max_sim_idx]['bboxes'].append(bbox)
        return matched, unmatched, untagged

    # Resets the current set of profiles
    def reset_current_profiles(self):
        self.profiles.clear()

src_model = DetectionModel(
    model_name='buffalo_l',
    model_root='./',
    det_size=(960, 960)
)

src_img_paths = [
    './assets/Source1.jpg',
    './assets/Source2.jpg',
    './assets/Source3.jpg'
]

# Distance threshold [0, 1] (lower is stricter)
dist_threshold = 0.4

# Collect embeddings and metadata
all_embs = []
# Keeps track of which image + bbox each embedding came from
all_meta = []

for path in src_img_paths:
    faces = src_model.get_group_faces(path)
    for face in faces:
        emb = np.ravel(face.normed_embedding)
        bbox = np.round(face.bbox).astype(np.int32)
        all_embs.append(emb)
        all_meta.append((path, bbox))

all_embs = np.stack(all_embs, axis=0)

# Compute cosine similarity -> clustering uses distance, so use (1 - similarity)
similarity_matrix = np.clip(cosine_similarity(all_embs), 0, 1)
distance_matrix = 1 - similarity_matrix

# Agglomerative clustering
clusterer = AgglomerativeClustering(
    metric='precomputed',
    linkage='average',
    distance_threshold=dist_threshold,
    n_clusters=None
)
labels = clusterer.fit_predict(distance_matrix)

# Group embeddings into profiles
profile_embs_list = []
src_bbox_data = []

for cluster_id in set(labels):
    indices = np.where(labels == cluster_id)[0]
    cluster_embs = [all_embs[i] for i in indices]
    profile_embs_list.append(cluster_embs)
    for i in indices:
        src_bbox_data.append(all_meta[i])

print(f'Detected {len(profile_embs_list)} unique identities in the source images.')

# Merging threshold [-1, 1] (higher is stricter)
merge_threshold = 0.6

# Return medoid (embedding closest to all others) from a cluster
def get_medoid(embs):
    sims = cosine_similarity(embs)
    medoid_idx = np.argmax(sims.sum(axis=1))
    return embs[medoid_idx]

# Merge clusters if their medoid embeddings are too similar
def merge_clusters(profile_embs_list, merge_threshold=0.75):
    # Compute medoids for each cluster
    medoids = np.array([get_medoid(embs) for embs in profile_embs_list])
    # Compute pairwise similarities between medoids
    sim_matrix = cosine_similarity(medoids)
    n = len(profile_embs_list)
    # Initially, each cluster is separate
    merged = [set([i]) for i in range(n)]
    # Iterative merging
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] > merge_threshold:
                # Merge cluster 'j' into cluster 'i'
                merged[i].update(merged[j])
                merged[j].clear()
    # Build new cluster list
    merged_clusters = []
    for group in merged:
        # Only keep non-empty sets
        if group:
            merged_embs = []
            for idx in group:
                merged_embs.extend(profile_embs_list[idx])
            merged_clusters.append(merged_embs)
    return merged_clusters

print(f'Before merging: {len(profile_embs_list)} profiles')

# Second pass: merge highly similar clusters
merged_profiles = merge_clusters(profile_embs_list, merge_threshold)

print(f'After merging: {len(merged_profiles)} profiles')

target_model = DetectionModel(
    model_name='buffalo_l',
    model_root='./',
    det_size=(1024, 1024)
)

identifier = ProfileIdentifier(target_model)

target_img_paths = [
    './assets/Target.jpg'
]

# Cosine threshold [-1, 1] (higher is stricter)
cos_threshold = 0.25

# Keeps track of the matched and unmatched identities
matched, unmatched, untagged = {}, {}, {}

# Reset the current profile list
identifier.reset_current_profiles()

# Create profiles for each individual from the generated profile embedding list
for i, profile_embs in enumerate(merged_profiles):
    identifier.create_profile(
        profile_embeddings=profile_embs,
        info={
            # ...
        }
    )

# Find matched identities in the target image(s)
for path in target_img_paths:
    matched[path], unmatched[path], untagged[path] = identifier.find_matches(path, cos_threshold)

import matplotlib.pyplot as plt

src_imgs = {}

for path in src_img_paths:
    src_imgs[path] = cv2.imread(path)
for path, bbox in src_bbox_data:
    cv2.rectangle(src_imgs[path], (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
for i, path in enumerate(src_img_paths):
    n_faces = sum([1 for p, bbox in src_bbox_data if p == path])
    print(f'Source #{i + 1}: ({n_faces} detected)')
    img = src_imgs[path]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

target_imgs = {}

for i, path in enumerate(target_img_paths):
    target_imgs[path] = cv2.imread(path)
    for profile in matched[path].values():
        for bbox in profile['bboxes']:
            cv2.rectangle(target_imgs[path], (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 3)
    for profile in unmatched[path].values():
        for bbox in profile['bboxes']:
            cv2.rectangle(target_imgs[path], (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
    for bbox in untagged[path]:
        cv2.rectangle(target_imgs[path], (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)

    n_matched = len(matched[path])
    n_unmatched = len(unmatched[path])
    n_untagged = len(untagged[path])
    n_total = n_matched + n_unmatched + n_untagged
    print(f'Target image #{i + 1}: ({n_matched} recognized / {n_total} detected) [Recognition: {n_matched / n_total * 100:.2f}%]')
    img = target_imgs[path]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()
