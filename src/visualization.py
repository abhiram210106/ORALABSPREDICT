"""
visualization.py
----------------
2D molecular rendering and 3D conformer generation with robust error handling
for OralAbsPredict.
"""

from typing import Optional, Tuple
import base64
from io import BytesIO
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D


def mol_to_2d_svg(
    mol: Chem.Mol,
    width: int = 400,
    height: int = 300,
    highlight_atoms: Optional[list] = None
) -> str:
    """
    Renders an RDKit Mol to a clean, scalable SVG string.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object.
    width : int
        Width of SVG canvas.
    height : int
        Height of SVG canvas.
    highlight_atoms : Optional[list]
        List of atom indices to highlight.

    Returns
    -------
    str
        SVG XML markup string.
    """
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        opts = drawer.drawOptions()
        opts.clearBackground = True
        opts.bondLineWidth = 2

        # Compute 2D coordinates if not already present
        if not mol.GetNumConformers():
            AllChem.Compute2DCoords(mol)

        if highlight_atoms:
            drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms)
        else:
            drawer.DrawMolecule(mol)

        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg
    except Exception as e:
        # Fallback to minimal SVG text message
        return (
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
            f'<text x="50%" y="50%" text-anchor="middle" fill="#888">2D Structure Error: {str(e)}</text>'
            f'</svg>'
        )


def mol_to_2d_png(
    mol: Chem.Mol,
    width: int = 400,
    height: int = 300
) -> Optional[Image.Image]:
    """
    Renders an RDKit Mol to a PIL PNG Image.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object.
    width : int
        Width of image.
    height : int
        Height of image.

    Returns
    -------
    Optional[Image.Image]
        PIL Image or None if rendering fails.
    """
    try:
        if not mol.GetNumConformers():
            AllChem.Compute2DCoords(mol)
        img = Draw.MolToImage(mol, size=(width, height))
        return img
    except Exception:
        return None


def mol_to_base64_png(mol: Chem.Mol, width: int = 400, height: int = 300) -> str:
    """
    Renders an RDKit Mol to a base64 encoded data URI for direct HTML display.
    """
    img = mol_to_2d_png(mol, width, height)
    if img is None:
        return ""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_b64}"


def generate_3d_conformer(
    mol: Chem.Mol,
    max_iters: int = 200,
    random_seed: int = 42
) -> Tuple[bool, Optional[str], Optional[Chem.Mol]]:
    """
    Generates a 3D conformer for an RDKit Mol using ETKDG and MMFF/UFF optimization.
    Never crashes on failure; returns gracefully.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object.
    max_iters : int, default 200
        Maximum optimization iterations.
    random_seed : int, default 42
        Reproducibility seed.

    Returns
    -------
    Tuple[bool, Optional[str], Optional[Chem.Mol]]
        (success, mol_block_3d, mol_with_conformer)
    """
    try:
        # Add hydrogens for accurate 3D geometry
        mol_h = Chem.AddHs(mol)

        # ETKDG v3 conformer generation
        params = AllChem.ETKDGv3()
        params.randomSeed = int(random_seed)

        embed_res = AllChem.EmbedMolecule(mol_h, params)
        if embed_res < 0:
            # Fallback to standard random coordinates embedding
            embed_res = AllChem.EmbedMolecule(mol_h, useRandomCoords=True, randomSeed=random_seed)

        if embed_res < 0:
            return False, None, None

        # Force-field optimization: try MMFF94, fallback to UFF
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol_h):
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=max_iters)
            else:
                AllChem.UFFOptimizeMolecule(mol_h, maxIters=max_iters)
        except Exception:
            pass  # Keep unoptimized conformer if forcefield fails

        # Generate 3D MolBlock (V2000 format)
        mol_block = Chem.MolToMolBlock(mol_h)
        return True, mol_block, mol_h

    except Exception:
        return False, None, None


def get_3dmol_html(mol_block: Optional[str], width: str = "100%", height: int = 350) -> str:
    """
    Generates standalone HTML with 3Dmol.js to render an interactive 3D molecule
    with rotate, zoom, and stick/cartoon representations.

    Parameters
    ----------
    mol_block : Optional[str]
        V2000 MolBlock string.
    width : str
        CSS width.
    height : int
        Height in pixels.

    Returns
    -------
    str
        HTML string with embedded 3Dmol.js viewer.
    """
    if not mol_block:
        return (
            f'<div style="width: {width}; height: {height}px; background: #0f172a; '
            f'display: flex; align-items: center; justify-content: center; '
            f'border-radius: 12px; color: #94a3b8; border: 1px dashed #334155;">'
            f'<span>3D Conformer not available for this structure</span></div>'
        )

    # Clean mol_block for JS string literal
    escaped_block = mol_block.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    html = f"""
    <div id="container-3dmol" style="width: {width}; height: {height}px; position: relative; border-radius: 12px; overflow: hidden; background: #0b1120; border: 1px solid #1e293b;">
        <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
        <div id="mol-viewer" style="width: 100%; height: 100%;"></div>
        <script>
            (function() {{
                try {{
                    let element = document.getElementById('mol-viewer');
                    let config = {{ backgroundColor: '#0b1120' }};
                    let viewer = $3Dmol.createViewer(element, config);
                    let molData = `{escaped_block}`;
                    viewer.addModel(molData, "mol");
                    viewer.setStyle({{}}, {{
                        stick: {{ radius: 0.15, colorscheme: 'Jmol' }},
                        sphere: {{ scale: 0.25, colorscheme: 'Jmol' }}
                    }});
                    viewer.zoomTo();
                    viewer.render();
                    viewer.spin(true);
                }} catch (e) {{
                    console.error("3Dmol render error:", e);
                }}
            }})();
        </script>
    </div>
    """
    return html
