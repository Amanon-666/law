"""Conditioned ReFi episodes from existing, correctly adapted nested S budgets."""
import torch
from recomposition_multibudget_natural import episode
from support_recomposition import attach_context


def conditioned_episode(parent, left, k):
    """Reuse H geometry; keep each budget's actual S fit and proposal density."""
    ep=episode(parent,k)
    payload=parent[1];state=payload['states'][k]
    u,gi=state['indices'],state['grid_indices']
    sx=payload['support_positions'][u]
    for ap,case in ep['cases'].items():
        block=case['blocks']
        if block['active']:
            block['left_ss']=left[ap]['left_ss'].index_select(1,u).index_select(2,u)
            block['left_gs']=left[ap]['left_gs'].index_select(1,gi).index_select(2,u)
            case['blocks']=attach_context(block,case['stats'],sx,ep['base']['length'])
        case['numerical_key']=f"{ep['id']}/K{k}/{ap}"
    ep['K']=k
    return ep
