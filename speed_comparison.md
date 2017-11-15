Comparing speeds
================
In this document we will compare the speed of various tasks, performed between networks of this library, of [pytorch-yolo2](https://github.com/marvis/pytorch-yolo2) and of [darknet](https://github.com/pjreddie/darknet).

## Single image test
This comparison was created by running a single image for inference through the network (Yolo-Voc with 1 class).  
To measurements were performed 25 times for each network.

| Measurement    | Statistic | Pytorch-Yolo2          | Lightnet                  | Darknet |
|:--------------:|:---------:|:----------------------:|:-------------------------:|:-------:|
|                | Minimum   | 0.342714786529541      | 0.3436274528503418        |  |
| Create network | Average   | 0.388584623336792      | __0.3805842685699463__    |  |
|                | Maximum   | 0.47118544578552246    | 0.4875929355621338        |  |
|                |           |                        |                           |  |
|                | Minimum   | 0.07985973358154297    | 0.07999420166015625       |  |
| Load weights   | Average   | 0.08387060165405273    | __0.08214405059814453__   |  |
|                | Maximum   | 0.11078214645385742    | 0.0931096076965332        |  |
|                |           |                        |                           |  |
|                | Minimum   | 0.003488779067993164   | 0.0031592845916748047     |  |
| Run network    | Average   | 0.026893777847290037   | __0.026136188507080077__  |  |
|                | Maximum   | 0.5856533050537109     | 0.5749838352203369        |  |
|                |           |                        |                           |  |
|                | Minimum   | 0.0027985572814941406  | 0.0016674995422363281     |  |
| Compute bbox   | Average   | 0.004696712493896484   | __0.004429044723510742__  |  |
|                | Maximum   | 0.005053997039794922   | 0.004963874816894531      |  |
|                |           |                        |                           |  |
|                | Minimum   | 3.3855438232421875e-05 | 3.3855438232421875e-05    |  |
| Perform nms    | Average   | 3.8003921508789064e-05 | __3.714561462402344e-05__ |  |
|                | Maximum   | 6.651878356933594e-05  | 7.224082946777344e-05     |  |
|                |           |                        |                           |  |
|                | Minimum   | 0.43095874786376953    | 0.43177199363708496       |  |
| TOTAL TIME     | Average   | 0.50408371925354       | __0.49333069801330565__   |  |
|                | Maximum   | 1.158980369567871      | 1.157426118850708         |  |

