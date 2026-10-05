from types import SimpleNamespace

import pytest

from frust.cluster import ClusterConfig, Resources
from frust.cluster.executor import update_executor, update_executor_with_dependencies


@pytest.mark.parametrize('dependent', [False, True])
def test_one_worker_is_explicit_even_for_one_cpu(dependent):
    captured = {}
    executor = SimpleNamespace(update_parameters=lambda **params: captured.update(params))
    cluster = ClusterConfig(partition='kemi1', extra_slurm_parameters={'account':'chemistry'})
    kwargs = {'dependency_job_ids':['123'],'dependency_type':'afterany'} if dependent else {}
    update = update_executor_with_dependencies if dependent else update_executor
    update(executor,cluster,Resources(1,2,5),job_name='fixture',**kwargs)
    assert captured['nodes']==1 and captured['slurm_ntasks_per_node']==1
    assert captured['cpus_per_task']==1
    assert captured['slurm_additional_parameters']['account']=='chemistry'
    from submitit.slurm.slurm import _make_sbatch_string
    script = _make_sbatch_string('python fixture.py','logs',nodes=captured['nodes'],
                                ntasks_per_node=captured['slurm_ntasks_per_node'],cpus_per_task=1)
    assert '#SBATCH --ntasks-per-node=1' in script
    assert '#SBATCH --nodes=1' in script


@pytest.mark.parametrize('flag', ['ntasks','ntasks_per_node','nodes','n','N'])
def test_multi_worker_overrides_rejected(flag):
    executor = SimpleNamespace(update_parameters=lambda **params: pytest.fail('Parameters were applied'))
    with pytest.raises(ValueError,match='one Python worker'):
        update_executor(executor,ClusterConfig(extra_slurm_parameters={flag:'2'}),
                        Resources(1,2,5),job_name='fixture')
