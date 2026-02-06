import numpy as np

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