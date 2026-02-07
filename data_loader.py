from pathlib import Path
import numpy as np

# Global Loader
def load_all_dataset_components(prm):
    """
    Loads all sketch and view data, and generates training pairs.
    """
    print("\n[INFO] Loading all dataset components...")

    # Load sketches and views
    (train_sketches, test_sketches, all_views,
     train_sketch_labels, test_sketch_labels, all_view_labels,
     label_names, class_name_to_id, model_to_class) = load_shrec13_data(prm)

    # Generate training pairs
    train_triples, train_labels = generate_pairs_shrec13(prm, train_sketch_labels, all_view_labels)

    # Data augmentation (only for training sketches if enabled)
    # The original load_data had data augmentation with .mat files.
    # If a new augmentation method is desired for the new PNG sketches, it should be implemented here.
    # For now, we'll assume augmentation is handled externally or not applied in this specific setup.
    # If prm.data_aug is True, you might add more sketches/labels here.
    # For simplicity, if data_aug is True but we don't have new data, it currently does nothing.
    # This might need revisiting if augmentation is critical and needs to be re-implemented for PNGs.
    if prm.data_aug:
        print("Data augmentation is enabled but currently no specific augmentation for PNGs is implemented here.")
        print("This needs to be added if image augmentation is desired for the new dataset structure.")


    return {
        'train_sketches': train_sketches,
        'test_sketches': test_sketches,
        'views': all_views, # Note: 'views' now contains all views, not separated by train/test for views
        'train_sketch_labels': train_sketch_labels,
        'test_sketch_labels': test_sketch_labels,
        'view_labels': all_view_labels, # All view labels
        'train_triples': train_triples,
        'train_labels': train_labels,
        'label_names': label_names,
        'class_name_to_id': class_name_to_id,
        'model_to_class': model_to_class
    }

# Triples Generator
def generate_pairs_shrec13(prm, sketch_labels, view_labels):
    """
    Genera coppie per training

    Formato triples: [sketch_idx, view_idx, sketch_peer_idx, view_peer_idx]
    Formato labels: [match_flag, sketch_class, view_class]

    - match_flag: 0 = coppia positiva (stessa classe), 1 = coppia negativa (classi diverse)
    - sketch_peer: sempre dalla stessa classe di sketch
    - view_peer: sempre dalla stessa classe di view
    """

    triples = []
    labels = []

    sketch_labels = sketch_labels.flatten()
    view_labels = view_labels.flatten()
    categories = np.unique(sketch_labels)

    max_pos = 50  # Max sketch per classe
    max_pos_view = 50  # Max view positive per classe
    max_neg_view = 50  # Max view negative

    posCt = 0
    negCt = 0

    for c in categories:
        # ===== SKETCH DELLA CLASSE CORRENTE =====
        inclass_sketch = np.where(sketch_labels == c)[0]
        if len(inclass_sketch) == 0:
            continue

        # Limita e mischia
        sketch_ind = np.arange(len(inclass_sketch))
        prm.rng.shuffle(sketch_ind)
        selected_sketch = inclass_sketch[sketch_ind[:min(max_pos, len(sketch_ind))]]

        # ===== VIEW DELLA CLASSE CORRENTE (per coppie positive) =====
        inclass_view = np.where(view_labels == c)[0]
        if len(inclass_view) == 0:
            continue

        # Limita e mischia
        view_ind = np.arange(len(inclass_view))
        prm.rng.shuffle(view_ind)
        selected_view = inclass_view[view_ind[:min(max_pos_view, len(view_ind))]]

        # ===== VIEW DI ALTRE CLASSI (per coppie negative) =====
        other_view = np.where(view_labels != c)[0]
        if len(other_view) == 0:
            continue

        # Limita e mischia
        view_neg_ind = np.arange(len(other_view))
        prm.rng.shuffle(view_neg_ind)
        other_view_selected = other_view[view_neg_ind[:min(max_neg_view, len(view_neg_ind))]]

        # ===== GENERA PEER PER POSITIVE =====
        # sketch_peer: sketch casuali dalla stessa classe c
        sketch_pos_peer = prm.rng.choice(inclass_sketch, size=len(selected_view), replace=True)

        # view_peer: view casuali dalla stessa classe c (mischiati)
        view_peer_ind = np.arange(len(selected_view))
        prm.rng.shuffle(view_peer_ind)
        view_pos_peer = selected_view[view_peer_ind]

        # ===== GENERA COPPIE POSITIVE =====
        for s in selected_sketch:
            # Limita a max 5 view positive per sketch
            for i in range(min(len(selected_view), 5)):
                v = selected_view[i]
                sp = sketch_pos_peer[i]
                vp = view_pos_peer[i]

                triples.append([s, v, sp, vp])
                labels.append([0, int(c), int(c)])  # match=0, entrambi classe c
                posCt += 1

        # ===== GENERA PEER PER NEGATIVE =====
        # Per ogni view negativa, serve sketch_peer e view_peer
        num_neg_pairs = len(other_view_selected)

        # sketch_peer: sketch casuali dalla classe c
        sketch_neg_peer = prm.rng.choice(inclass_sketch, size=num_neg_pairs, replace=True)

        # view_peer: per ogni view negativa, sceglie un peer dalla STESSA classe della view
        view_neg_peer = []
        for v_neg in other_view_selected:
            v_class = view_labels[v_neg]
            v_class_views = np.where(view_labels == v_class)[0]
            if len(v_class_views) > 0:
                vp = prm.rng.choice(v_class_views)
                view_neg_peer.append(vp)
            else:
                view_neg_peer.append(v_neg)  # Fallback

        # ===== GENERA COPPIE NEGATIVE =====
        for s in selected_sketch:
            # Limita a max 10 view negative per sketch
            for i in range(min(len(other_view_selected), 10)):
                v = other_view_selected[i]
                sp = sketch_neg_peer[i]
                vp = view_neg_peer[i]
                v_class = view_labels[v]

                triples.append([s, v, sp, vp])
                labels.append([1, int(c), int(v_class)])  # match=1, classi diverse
                negCt += 1

    triples = np.array(triples)
    labels = np.array(labels)

    # Shuffle finale
    idx = np.arange(len(triples))
    prm.rng.shuffle(idx)

    print(f"Generated {len(triples)} pairs ({posCt} positive, {negCt} negative)")

    return triples[idx], labels[idx]

# Global Data Loader
def load_shrec13_data(prm):
    """
    Carica sketch PNG per train e test e genera view da modelli 3D

    Returns:
        train_sketches: array (N_train, 10000) - immagini train 100×100 flatten
        test_sketches: array (N_test, 10000) - immagini test 100×100 flatten
        views: array (M, 10000) - view renderizzate 100×100 flatten
        train_sketch_labels: array (N_train, 1) - classe di ogni sketch train
        test_sketch_labels: array (N_test, 1) - classe di ogni sketch test
        view_labels: array (M, 1) - classe di ogni view
        label_names: list - nomi delle categorie (indice = class_id)
        class_name_to_id: dict - mappa nome_categoria → class_id
        model_to_class: dict - mappa model_id → class_id
    """
    global label_names, class_name_to_id, model_to_class

    # Initialize if not already initialized (e.g., if run in isolation)
    if 'label_names' not in globals():
        label_names = []
    if 'class_name_to_id' not in globals():
        class_name_to_id = {}
    if 'model_to_class' not in globals():
        model_to_class = {}

    # Path delle directory
    train_sketch_dir = Path('/content/train_schetch')
    test_sketch_dir = Path('/content/test_schetch')
    views_dir = Path('/content/views')

    # ===== BUILD LABEL NAMES MAPPING =====
    class_file = Path('/content/evaluation/SHREC13_SBR_Model.cla')


    if class_file.exists():
        with open(class_file, 'r') as f:
            lines = f.readlines()

        # Salta le prime 2 righe (header)
        i = 2
        current_class_id = -1
        num_models_remaining = 0

        while i < len(lines):
            line = lines[i].strip()

            # Salta righe vuote o solo spazi
            if not line:
                i += 1
                continue

            parts = line.split()

            # Se la riga ha 3 parti: category_name 0 num_models
            if len(parts) == 3 and parts[1] == '0':
                category_name = parts[0]  # es: "airplane"
                num_models = int(parts[2])  # es: 184

                # Assegna un class_id progressivo
                current_class_id = len(label_names)
                label_names.append(category_name)
                class_name_to_id[category_name.lower()] = current_class_id

                num_models_remaining = num_models

            # Altrimenti è un model_id
            elif len(parts) == 1 and parts[0].isdigit():
                model_id = parts[0]
                if current_class_id >= 0 and num_models_remaining > 0:
                    model_to_class[model_id] = current_class_id
                    num_models_remaining -= 1

            i += 1
    else:
        raise FileNotFoundError(f"Class file not found: {class_file}")

    # ===== CARICA SKETCH TRAIN E TEST =====
    train_sketches, train_sketch_labels = load_sketches_from_dir(train_sketch_dir, prm)

    test_sketches, test_sketch_labels = load_sketches_from_dir(test_sketch_dir, prm)

    views_list, view_labels_list = load_views_from_dir(views_dir, prm)

    views = np.array(views_list)
    view_labels = np.array(view_labels_list).reshape(-1, 1)

    return train_sketches, test_sketches, views, train_sketch_labels, test_sketch_labels, view_labels, label_names, class_name_to_id, model_to_class

# FUNCTION TO LOAD VIEWS FROM A DIRECTORY
def load_views_from_dir(views_dir, prm):
    views_list = []
    view_labels_list = [] # Initialize view_labels_list

    # Carica tutti i PNG nella cartella
    png_files = sorted(views_dir.glob('*.png'))

    for png_file in png_files:
        # Leggi immagine in grayscale
        img = cv2.imread(str(png_file), cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        shape = img.shape

        if shape[0] != prm.inputWH or shape[1] != prm.inputWH:
          img = cv2.resize(img, (prm.inputWH, prm.inputWH))
        # controllare meglio il resizing se non è troppo

        # Normalizza [0, 255] → [0, 1]
        img = img.astype(np.float32) / 255.0

        # Flatten
        img_flat = img.flatten()

        views_list.append(img_flat)

        # Corrected line: Use png_file.name to get the string representation
        model_id = png_file.name.split('_')[0]

        # Remove initial "m" if present
        if model_id.startswith('m'):
            model_id = model_id[1:]

        view_labels_list.append(model_to_class[model_id])

    views = np.array(views_list)
    views_labels = np.array(view_labels_list).reshape(-1, 1)

    return views, views_labels

# FUNCTION TO LOAD SKETCH FROM A DIRECTORY
def load_sketches_from_dir(sketch_dir, prm):
    sketches_list = []
    sketch_labels_list = []

    # Itera sulle cartelle di classe
    for class_folder in sorted(sketch_dir.iterdir()):
        if not class_folder.is_dir():
            continue

        class_name = class_folder.name.lower()  # es: "airplane"

        # Ottieni class_id dalla mappa
        class_id = class_name_to_id.get(class_name, -1)

        if class_id == -1:
            print(f"Warning: categoria '{class_name}' non trovata in .cla file")
            continue

        # Carica tutti i PNG nella cartella
        png_files = sorted(class_folder.glob('*.png'))

        for png_file in png_files:
            # Leggi immagine in grayscale
            img = cv2.imread(str(png_file), cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            # Resize a 100×100
            img = cv2.resize(img, (prm.inputWH, prm.inputWH))
 # controllare meglio il resizing se non è troppo

            # Normalizza [0, 255] → [0, 1]
            img = img.astype(np.float32) / 255.0

            # Flatten
            img_flat = img.flatten()

            sketches_list.append(img_flat)
            sketch_labels_list.append(class_id)

    sketches = np.array(sketches_list)
    sketch_labels = np.array(sketch_labels_list).reshape(-1, 1)

    return sketches, sketch_labels