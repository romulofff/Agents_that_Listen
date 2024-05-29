import json

# Carregar os dados do arquivo json
with open('train_dir/curriculum_3/actor_1/actor_1_vis_som_caso3_mais_30/wall_challenge_2_actor_1_vis_som_caso3_mais_30_counter.json') as f:
# with open('train_dir/curriculum_3/actor_full/actor_full_curr_c3/wall_challenge_2_actor_full_curr_c3_counter.json') as f:
# with open('train_dir/curriculum_3/actor_full/actor_vision_only_c3/wall_challenge_2_actor_vision_only_c3_counter.json') as f:
    data = json.load(f)

# Contar a frequência de cada episódio
episode_counts = {}
for item in data:
    episode = item['episode']
    if episode in episode_counts:
        episode_counts[episode] += 1
    else:
        episode_counts[episode] = 1

# Contar quantos episódios aparecem duas vezes
double_episodes = sum(1 for count in episode_counts.values() if count == 2)

print("Captured both medikits: ",double_episodes)

# print(episode_counts)

# Armazenar a primeira aparição de cada episódio
first_appearance = {}

# Contar quantas vezes a "diff" da segunda aparição do episódio é menor do que 60
counter = 0
for item in data:
    episode = item['episode']
    diff = item['diff']
    if episode in first_appearance:
        if abs(diff - first_appearance[episode]) < 55:
            counter += 1
    else:
        first_appearance[episode] = diff

print("Diff smaller than 55:", counter)

# print(first_appearance)