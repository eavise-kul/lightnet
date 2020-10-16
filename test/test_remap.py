#
#   Test if network weight remapping works
#   Copyright EAVISE
#

import inspect
import pytest
import torch
import lightnet as ln

remaps = [
    # source,                   target,                             remap
    (ln.models.Darknet,         ln.models.TinyYoloV2,               ln.models.TinyYoloV2.remap_darknet),
    (ln.models.Darknet,         ln.models.TinyYoloV3,               ln.models.TinyYoloV3.remap_darknet),
    (ln.models.Darknet19,       ln.models.DYolo,                    ln.models.DYolo.remap_darknet19),
    (ln.models.Darknet19,       ln.models.Yolt,                     ln.models.Yolt.remap_darknet19),
    (ln.models.Darknet19,       ln.models.YoloV2,                   ln.models.YoloV2.remap_darknet19),
    (ln.models.Darknet19,       ln.models.YoloV2Upsample,           ln.models.YoloV2Upsample.remap_darknet19),
    (ln.models.Darknet53,       ln.models.YoloV3,                   ln.models.YoloV3.remap_darknet53),
    (ln.models.MobileDarknet19, ln.models.MobileYoloV2,             ln.models.MobileYoloV2.remap_mobile_darknet19),
    (ln.models.MobileDarknet19, ln.models.MobileYoloV2Upsample,     ln.models.MobileYoloV2.remap_mobile_darknet19),
    (ln.models.MobilenetV1,     ln.models.MobilenetYolo,            ln.models.MobilenetYolo.remap_mobilenet_v1),
]

# Difficult to test (usually remaps from other repos)
remap_skips = [
    ln.models.Cornernet.remap_princeton_vl,
    ln.models.CornernetSqueeze.remap_princeton_vl,
]


@pytest.mark.parametrize('remap', remaps)
def test_remapping(remap, tmp_path):
    # Create networks
    source = remap[0](1000)
    target = remap[1](20)

    # Save weights
    weight_file = str(tmp_path / 'weights.pt')
    source.save(weight_file, remap=remap[2])

    # Check that there are only missing layers and no wrong layers in weight file
    weight_keys = torch.load(weight_file, 'cpu').keys()
    target_keys = target.state_dict().keys()
    assert len(set(weight_keys) - set(target_keys)) == 0

    # Check if loading works
    target.load(weight_file, strict=False)


# All remaps tested?
def test_all_remaps_tested():
    networks = [
        getattr(ln.models, net) for net in dir(ln.models)
        if (inspect.isclass(getattr(ln.models, net)))
        and (issubclass(getattr(ln.models, net), torch.nn.Module))
    ]

    for net in networks:
        net_remaps = [(r, getattr(net, r)) for r in dir(net) if r.startswith('remap')]
        tested_remaps = [r[2] for r in remaps if r[1] == net]

        for remap in net_remaps:
            if remap[1] in remap_skips:
                continue
            if remap[1] not in tested_remaps:
                raise NotImplementedError(f'Remap [{remap[0]}] of Network [{net.__name__}] is not being tested!')
