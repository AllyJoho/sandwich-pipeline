"""Maya's side of assembly decomposition: Split Pieces, Edit Piece, Save Piece.

- A group's name is the whole declaration: `<asset>` or `<asset>__<variant>`.
- A split variant has no textures of its own: its materials read the
  assembly's `publish/tex/main`, found through the stamp on its source layer.
- A split variant is modelled only from inside the assembly. A child may also
  have a `model.mb` for variants of its own; a split never joins any other
  asset modelled by hand.
- A piece is placed by moving its prim in the assembly, never its open copy.
"""
