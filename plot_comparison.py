import pickle
import os
from evaluate_roberta import get_roberta_score
import numpy as np
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

def create_finalout(file_path, file_index):
    vrnn_data_dict = pickle.load(open('plots/output_dicts/vrnn/12_vrnn_GRU_sentence.pkl', 'rb'))
    bi_vrnn_data_dict = pickle.load(open('plots/output_dicts/bivrnn/72_bivrnn_GRU_sentence.pkl', 'rb'))
    # bi_vrnn_data_dict = pickle.load(open(file_path, 'rb'))
    
    kld = {'vrnn' : [], 'bivrnn': []}
    rec = {'vrnn' : [], 'bivrnn': []}
    loss = {'vrnn' : [], 'bivrnn': []}
    score = {'vrnn' : [[None], [None]], 'bivrnn': [[None], [None]]}
    for epoch in tqdm(range(100)):
        temp_vrnn = vrnn_data_dict[f'epoch{epoch+1}']
        temp_bivrnn = bi_vrnn_data_dict[f'epoch{epoch+1}']

        kld['vrnn'].append(temp_vrnn['kld'])
        kld['bivrnn'].append(temp_bivrnn['kld'])

        rec['vrnn'].append(temp_vrnn['rec'])
        rec['bivrnn'].append(temp_bivrnn['rec'])

        loss['vrnn'].append(temp_vrnn['loss'])
        loss['bivrnn'].append(temp_bivrnn['loss'])

        while True:
                try:
                    # vrnnscores = np.mean(np.array(get_roberta_score(list(np.array(temp_vrnn['sample'])[:, 0]))), axis=0)
                    bivrnnscores = np.mean(np.array(get_roberta_score(list(np.array(temp_bivrnn['sample'])[:, 0]))), axis=0)
                    break
                except:
                    continue
        # score['vrnn'][0].append(vrnnscores[0])
        # score['vrnn'][1].append(vrnnscores[1])
        score['bivrnn'][0].append(bivrnnscores[0])
        score['bivrnn'][1].append(bivrnnscores[1])

    final_out_dict = {'kld': kld, 'rec': rec, 'loss': loss, 'score': score}
    with open(f"plots/output_dicts/{file_index}_final_out_dict.pkl", 'wb') as file:
            pickle.dump(final_out_dict, file)

def create_plot_from_final_out(file_index):
    final_data = pickle.load(open(f'plots/output_dicts/{file_index}_final_out_dict.pkl', 'rb'))
    kld = final_data['kld']
    rec = final_data['rec']
    loss = final_data['loss']
    scores = final_data['score']

    plt.figure(dpi=300)
    plt.axhline(np.mean(kld['vrnn']), color='orange', linestyle='--', alpha=0.5, linewidth=0.8)
    plt.axhline(np.mean(kld['bivrnn']), color='blue', linestyle='--', alpha=0.5, linewidth=0.8)
    plt.plot(kld['vrnn'], label="VRNN", color='orange')
    plt.loglog(kld['bivrnn'], label="BI_VRNN", color='blue')
    plt.xlabel('Number of Epochs')
    plt.ylabel('KL Divergence')
    plt.legend()
    plt.title('KL Divergence vs Number of Epochs')
    plt.savefig(f'plots/images/{file_index}_kld.png')

    plt.figure(dpi=300)
    plt.axhline(np.mean(rec['vrnn']), color='orange', linestyle='--', alpha=0.5, linewidth=0.8)
    plt.axhline(np.mean(rec['bivrnn']), color='blue', linestyle='--', alpha=0.5, linewidth=0.8)
    plt.plot(rec['vrnn'], label="VRNN", color='orange')
    plt.loglog(rec['bivrnn'], label="BI_VRNN", color='blue')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Reconstruction Loss')
    plt.legend()
    plt.title('Reconstruction Loss vs Number of Epochs')
    plt.savefig(f'plots/images/{file_index}_rec.png')

    plt.figure(dpi=300)
    plt.axhline(np.mean(loss['vrnn']), color='orange', linestyle='--', alpha=0.5, linewidth=0.8)
    plt.axhline(np.mean(loss['bivrnn']), color='blue', linestyle='--', alpha=0.5, linewidth=0.8)
    plt.plot(loss['vrnn'], label="VRNN", color='orange')
    plt.loglog(loss['bivrnn'], label="BI_VRNN", color='blue')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.title('Total Loss vs Number of Epochs')
    plt.savefig(f'plots/images/{file_index}_loss.png')

    plt.figure(dpi=300)
    # plt.axhline(np.mean(scores['vrnn'][0][1:]), color='orange', linestyle='--', alpha=0.5, linewidth=0.8)
    plt.axhline(np.mean(scores['bivrnn'][0][1:]), color='blue', linestyle='--', alpha=0.5, linewidth=0.8)
    # plt.plot(scores['vrnn'][0][1:], label="VRNN", color='orange')
    plt.loglog(scores['bivrnn'][0][1:], label="BI_VRNN", color='blue')
    plt.xlabel('Number of Epochs')
    plt.ylabel('score_0')
    plt.legend()
    plt.title('score_0 vs Number of Epochs')
    plt.savefig(f'plots/images/{file_index}_score_0.png')

    plt.figure(dpi=300)
    # plt.axhline(np.mean(scores['vrnn'][1][1:]), color='orange', linestyle='--', alpha=0.5, linewidth=0.8)
    plt.axhline(np.mean(scores['bivrnn'][1][1:]), color='blue', linestyle='--', alpha=0.5, linewidth=0.8)
    plt.plot(scores['vrnn'][1][1:], label="VRNN", color='orange')
    plt.plot(scores['bivrnn'][1][1:], label="BI_VRNN", color='blue')
    plt.xlabel('Number of Epochs')
    plt.ylabel('score_1')
    plt.legend()
    plt.title('score_1 vs Number of Epochs')
    plt.savefig(f'plots/images/{file_index}_score_1.png')


# print(vrnn_data_dict['epoch30']['loss'])
# print(bi_vrnn_data_dict['epoch30']['loss'])
# print(list(np.array(data_dict['epoch1']['sample'])[:, 0]))
# print(get_roberta_score(list(np.array(data_dict['epoch1']['sample'])[:, 0])))
# for sample in data_dict['epoch1']['sample']:
#     print(get_roberta_score(sample[0]))

def get_file_index():
    # Get integer for output file name
    output_dir = 'plots/images/'
    largest_integer = 0
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        
        for file in files:
            if file.startswith(tuple(map(str, range(10)))):
                # Extract the integer from the file name
                file_integer = int(file.split('_')[0])
                
                # Update the largest integer if necessary
                if file_integer > largest_integer:
                    largest_integer = file_integer
        
        # Increment the largest integer by 1
        largest_integer += 1

    return largest_integer

create_final_dict = True
# file_index = get_file_index()
file_path = 'plots/output_dicts/bivrnn/72_bivrnn_GRU_sentence.pkl'
file_index = "72"
if create_final_dict:
    create_finalout(file_path, file_index)
    create_plot_from_final_out(file_index)
else:
    create_plot_from_final_out(file_index)