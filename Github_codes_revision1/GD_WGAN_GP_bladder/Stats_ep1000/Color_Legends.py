import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Your color array
color_array = [
    '#0173b2','#0173b2','#0173b2',
    '#de8f05','#de8f05','#de8f05','#de8f05','#de8f05','#de8f05',
    '#029e73','#029e73','#029e73','#029e73','#029e73',
    '#cc78bc','#cc78bc','#cc78bc',
    '#949494','#949494','#949494','#949494'
]

# Deduplicate colors and assign labels
unique_colors = {
    'Baselines': '#0173b2',
    'Compared method - MolGAN': '#de8f05',
    'Proposed methods with CNN generator': '#029e73',
    'Ablation on GD-WGAN-GP': '#cc78bc',
    'Proposed methods with\nLSTM and Transformer generator': '#949494'
}

# Make legend handles
handles = [Patch(facecolor=c, label=lbl) for lbl, c in unique_colors.items()]

# Plot only the legend
plt.figure()
plt.legend(handles=handles, title="Groups", loc="center")
plt.axis("off")  # hide axes
plt.savefig('Method_box_legends.png')
plt.show()
