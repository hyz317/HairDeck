"""Packed card templates (*_templates*.npz): every card of every hairstyle in flat arrays.

points (P, 5) float32 = xyz, width, thickness per control point; card_ptr (C+1,) offsets into points; card_shape (C,)
uint8 (0 square, 1 triangle); hair_ptr (H+1,) offsets into the cards; card_nrm (C, 3) float32, the root width direction
of each card (absent in CHARM's hair.json templates).
"""
import numpy as np


class Packed:
    def __init__(self, path):
        z = np.load(path)
        self.points, self.card_ptr, self.card_shape, self.hair_ptr = z['points'], z['card_ptr'], z['card_shape'], z['hair_ptr']
        self.nrm = z['card_nrm'] if 'card_nrm' in z.files else None

    def __len__(self):
        return len(self.hair_ptr) - 1

    def cards(self, i):
        """(seq (n,8), shape_id) with the root normal appended to every row; (seq (n,5), shape_id) for a file without
        card_nrm (CHARM's hair.json templates)."""
        out = []
        for c in range(self.hair_ptr[i], self.hair_ptr[i + 1]):
            s = self.points[self.card_ptr[c]:self.card_ptr[c + 1]].astype(np.float64)
            if self.nrm is not None:
                s = np.concatenate([s, np.repeat(self.nrm[c][None].astype(np.float64), len(s), 0)], 1)
            out.append((s, int(self.card_shape[c])))
        return out
