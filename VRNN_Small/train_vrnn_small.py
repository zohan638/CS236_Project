def train(model, data_loader, num_epochs):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
        total_loss = 0
        hidden = None

        for x_batch in data_loader:  
            x_batch = Variable(x_batch) 

            # Forward pass
            loss1, loss2, hidden = model.calculate_loss(x_batch, hidden)
            loss = loss1 + loss2  # Total loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.data

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(data_loader)}')
    sample_string = model.sample(20, 'obama')
    print(f"Sample: {sample_string}")

if __name__ == '__main__':
    args = parse_arguments()

    #Initialisations
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    torch.cuda.manual_seed(args['seed'])
    torch.backends.cudnn.deterministic = True
    args['device'] = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))

    # Load padded sentences
    padded_sentences = np.load('data/vrnn_padded_sentences.npy')

    # Convert numpy array to PyTorch tensor
    padded_sentences_tensor = torch.from_numpy(padded_sentences).long()
    
    # Load vocabulary
    with open('data/vrnn_vocabulary.json', 'r') as f:
        vocab = json.load(f)

    # Create data loader
    dataset = TextDataset(padded_sentences_tensor)
    data_loader = DataLoader(dataset, batch_size=200, shuffle=False)  # Adjust batch size as needed

    preset_args = {
        'vocab': vocab,
        'embedding_size': 24,
        'h_dim': 256,
        'z_dim': 24,
        'n_layers': 2,
        'bias': False,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }

    # Create model
    model = VRNN(preset_args)
    model.to(args['device'])
    train_model(model, data_loader, epochs=10)