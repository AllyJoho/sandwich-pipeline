"""Maya's side of assembly decomposition: Split Pieces, Edit Piece, Save Piece.

- A group's name is the whole declaration: `<asset>` or `<asset>__<variant>`.
- Splitting runs on Linux only: a child's `publish/tex/<variant>` is a folder
  of symlinks to the assembly's textures, and Windows cannot make them.
- A split variant is modelled only from inside the assembly. A child may also
  have a `model.mb` for variants of its own; a split never joins any other
  asset modelled by hand.
- A piece is placed by moving its prim in the assembly, never its open copy.
"""
