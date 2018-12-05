### Solarisation :waning_crescent_moon:
 Solarisation refers to the effect that the optical density decreases with correspondingly increasing exposure. Non-linear grey value transformations.
____

This effect can be reached by a non-linear transformation of the density curve. Gray values mapping y(x) to the polynomial of the third degree.
- Take two parameters (x0, y0) and (x1, y1) and compute a polynomial of third degree
- Apply polynomial to the image
___
#### Result
1. Original
<img src="./dog.jpg" height="250"> 

2. `10, 20, 180, 255`
<img src="./10_20_180_255.jpg" height="250"> 

3. `0, 255, 255, 0`
<img src="./0_255_255_0.jpg" height="250"> 
