import numpy as np
import pickle
import os
import time
import tensorflow as tf

# 3. Training CNN RNN model to generate the captions for image.

######################################## Training Data Generation ############################################
# load the training images names
with open('Flicker8k_text/Flickr_8k.trainImages.txt') as file:
    photos = file.read().split('\n')[:-1]


def load_clean_captions(filename, photos): 
    #loading clean_captions
    with open(filename) as f:
        file = f.read().split('\n')
        
    descriptions = {}
    for line in file:
        words = line.split()
        if len(words)<1 :
            continue
        image, image_caption = words[0], words[1:]
        if image in photos:
            if image not in descriptions:
                descriptions[image] = []
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)
    return descriptions


def load_features(photos):
    #loading all features
    all_features = pickle.load(open("features.pkl","rb"))
    #selecting only needed features
    features = {image:all_features[image] for image in photos}
    return features


# preparing training data
filename = 'Flicker8k_text/Flickr_8k.trainImages.txt'
train_imgs = photos
train_captions = load_clean_captions("captions.txt", train_imgs)
train_features = load_features(train_imgs)


def dict_to_list(caption_dict):
    all_caps = []
    
    for key in caption_dict.keys():
        [all_caps.append(d) for d in caption_dict[key]]
        
    return all_caps


def create_tokenizer(caption_dict):
    caps_list = dict_to_list(caption_dict)
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(caps_list)
    return tokenizer

# craeting tokenizer object
tokenizer = create_tokenizer(train_captions)
pickle.dump(tokenizer, open('tokenizer.pkl', 'wb'))
vocab_size = len(tokenizer.word_index) + 1

max_length = max(len(c.split()) for c in dict_to_list(train_captions))


def create_sequences(tokenizer, max_length, caps_list, feature):
    X1, X2, y = list(), list(), list()
    # walk through each caption for the image
    for caps in caps_list:
        # encode the sequence
        seq = tokenizer.texts_to_sequences([caps])[0]
        # split one sequence into multiple X,y pairs
        for i in range(1, len(seq)):
            # split into input and output pair
            in_seq, out_seq = seq[:i], seq[i]
            # pad input sequence
            in_seq = tf.keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=max_length)[0]
            # encode output sequence
            out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
            # store
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)
            
    return np.array(X1), np.array(X2), np.array(y)


#create input-output sequence pairs from the image description.

#data generator, used by model.fit_generator()
def data_generator(captions, features, tokenizer, max_length):
    while 1:
        for key, captions_list in captions.items():
            #retrieve photo features
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, captions_list, feature)
            yield [[input_image, input_sequence], output_word]


[a,b],c = next(data_generator(train_captions, train_features, tokenizer, max_length))

#############################################################################################################


######################################## Training RNN CNN model #############################################

# define the cnn-rnn captioning model
def cnn_rnn_model(vocab_size, max_length):

    # features from the CNN model squeezed from 2048 to 256 nodes
    inputs1 = tf.keras.layers.Input(shape=(2048,))
    fe1 = tf.keras.layers.Dropout(0.5)(inputs1)
    fe2 = tf.keras.layers.Dense(256, activation='relu')(fe1)

    # LSTM sequence model
    inputs2 = tf.keras.layers.Input(shape=(max_length,))
    se1 = tf.keras.layers.Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = tf.keras.layers.Dropout(0.5)(se1)
    se3 = tf.keras.layers.LSTM(256)(se2)

    # Merging both models
    decoder1 = tf.keras.layers.add([fe2, se3])
    decoder2 = tf.keras.layers.Dense(256, activation='relu')(decoder1)
    outputs = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder2)

    # tie it together [image, seq] [word]
    model = tf.keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # summarize model
    print(model.summary())
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    return model


# train our model
print('Dataset: ', len(train_imgs))
print('Descriptions: train=', len(train_captions))
print('Photos: train=', len(train_features))
print('Vocabulary Size:', vocab_size)
print('Description Length: ', max_length)

start_time = time.time()

model = cnn_rnn_model(vocab_size, max_length)

epochs = 30
steps = len(train_captions)

# making a directory models to save our models
os.mkdir("models")

for i in range(epochs):
    generator = data_generator(train_captions, train_features, tokenizer, max_length)
    model.fit_generator(generator, epochs=1, steps_per_epoch= steps, verbose=1)
    model.save("models/model_" + str(i) + ".h5")
    
end_time = time.time()

print(f"total time taken by model to train is : {end_time - start_time}")

################################################### END #######################################################