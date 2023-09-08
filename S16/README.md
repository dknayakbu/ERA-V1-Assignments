# ERA-V1-Assignments for Session 16
This repository contains all the ERA V1 session 16 Assignments.

In this assignment, we will speed up the training process by efficiently training the transformers.

The task at hand is to reach a loss of less than 1.8 on the English-French translation dataset.

We will be implementing the below:

## OneCycle Policy
The One Cycle Policy is a technique used in deep learning to train complex models faster and with fewer iterations. It follows the Cyclical Learning Rate (CLR) to obtain faster training time with a regularization effect but with a slight modification. Specifically, it uses one cycle that is smaller than the total number of iterations/epochs and allows the learning rate to decrease several orders of magnitude less than the initial learning rate for the remaining iterations (i.e. last few iterations)


## Dynamic Padding
Dynamic padding is a technique used in natural language processing to optimize the padding process during batch creation. Instead of padding all the samples to the maximum length, dynamic padding limits the number of added pad tokens to reach the length of the longest sequence of each mini-batch. This technique is more efficient than traditional padding because it reduces the amount of unnecessary padding, which speeds up training. 


## Automatic Mixed Precision
Automatic Mixed Precision (AMP) is a technique used in deep learning to speed up training and reduce memory usage by combining different numerical formats in one computational workload. AMP is supported by popular deep learning frameworks such as TensorFlow, PyTorch, and MXNet. In the context of Transformers, AMP is used to train models faster by training data in a half-precision floating point (FP16) compared to a single-precision floating point (FP32). AMP is similar to FP16 mixed precision, but it uses both single and half-precision representations. AMP automates mixed precision by using a combination of automatic casting and scaling of gradients


## Parameter Sharing
Parameter sharing is a technique used in Transformers to reduce the number of parameters in the model and improve its efficiency. In general, parameter sharing involves using the same set of parameters for multiple layers of the model. There are several ways to implement parameter sharing in Transformers, including:
    - Sharing parameters for one layer with all layers, as in Universal Transformers.
    - Repeating the entire Transformer layer a given number of times, as in classic (ALBERT-like) sharing.
    - Using sequence, cycle, or cycle (rev) strategies to perform weight sharing on Transformer models

# Training Results and Log
```
Using device: cuda
Max length of source sentence: 471
Max length of target sentence: 482
Processing Epoch 00: 100%|████████████████████████████████████████████████████████████| 1929/1929 [02:06<00:00, 15.23it/s, loss=4.684]
stty: 'standard input': Inappropriate ioctl for device
Average loss of epoch 0 is 5.737465578883098
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
--------------------------------------------------------------------------------
    SOURCE: "Brother? Yes; at the distance of a thousand leagues! Sisters? Yes; slaving amongst strangers! I, wealthy--gorged with gold I never earned and do not merit! You, penniless! Famous equality and fraternisation! Close union! Intimate attachment!"
    TARGET: -- Mon frère éloigné de mille lieues, mes soeurs asservies chez des étrangers, et moi riche, gorgée d'or, sans l'avoir jamais ni gagné ni mérité! Est-ce là une égalité fraternelle, une union ultime, un profond attachement?
 PREDICTED: -- Et , dit le bout de la petite petite petite petite petite petite petite !
--------------------------------------------------------------------------------
    SOURCE: I can see him now, for he had so deep a crease across his brown cheek that no tear could pass it, but must trickle away sideways and so down to his ear, hopping off on to the sheet of paper.
    TARGET: Je le voyais fort bien à présent, car il avait à travers sa joue pâlie une ride si profonde, qu'aucune larme ne pouvait la franchir. Il fallait qu'elle glissât de côté jusqu'à son oreille, d'où elle tombait sur la feuille de papier.
 PREDICTED: Je ne sais que le , il avait été une femme de sa femme , mais il ne me pas à son coeur .
--------------------------------------------------------------------------------
Processing Epoch 01: 100%|████████████████████████████████████████████████████████████| 1929/1929 [02:07<00:00, 15.11it/s, loss=4.297]
stty: 'standard input': Inappropriate ioctl for device
Average loss of epoch 1 is 4.392081091106093
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
--------------------------------------------------------------------------------
    SOURCE: The next day the sky was again overcast; but on the 29th of June, the last day but one of the month, with the change of the moon came a change of weather.
    TARGET: Le lendemain le ciel fut encore couvert, mais le dimanche, 28 juin, l'antépénultième jour du mois, avec le changement de lune vint le changement de temps.
 PREDICTED: Le lendemain , le ciel était encore , mais au milieu des mois , le lendemain , le jour , la pluie de la lune , la pluie d ’ une grande nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle nouvelle .
--------------------------------------------------------------------------------
    SOURCE: "A husband for Cousin Edie," said I.
    TARGET: -- Un mari pour la cousine Edie, répondis-je.
 PREDICTED: -- Un mari pour Edie , dis - je .
--------------------------------------------------------------------------------
Processing Epoch 02: 100%|████████████████████████████████████████████████████████████| 1929/1929 [02:07<00:00, 15.18it/s, loss=3.814]
stty: 'standard input': Inappropriate ioctl for device
Average loss of epoch 2 is 3.8029715834410465
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
--------------------------------------------------------------------------------
    SOURCE: I agree, therefore, with Mr. Spilett, that she must be left in Port Balloon.
    TARGET: Je pense donc, comme M Spilett, qu'il faut le laisser à port-ballon.
 PREDICTED: Je suis donc à l ' avis de monsieur Spilett , qu ' elle devait quitter le port ballon .
--------------------------------------------------------------------------------
    SOURCE: I cannot find in that single circumstance a reason for admiration.
    TARGET: Je ne puis trouver dans cette seule circonstance une raison pour admirer.
 PREDICTED: Je ne puis trouver une raison pour l ’ admiration .
--------------------------------------------------------------------------------
Processing Epoch 03: 100%|████████████████████████████████████████████████████████████| 1929/1929 [02:07<00:00, 15.10it/s, loss=3.354]
stty: 'standard input': Inappropriate ioctl for device
Average loss of epoch 3 is 3.459429015020684
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
--------------------------------------------------------------------------------
    SOURCE: The respect created by the conviction of his valuable qualities, though at first unwillingly admitted, had for some time ceased to be repugnant to her feeling; and it was now heightened into somewhat of a friendlier nature, by the testimony so highly in his favour, and bringing forward his disposition in so amiable a light, which yesterday had produced.
    TARGET: Depuis quelque temps déja elle avait cessé de lutter contre le respect que lui inspiraient ses indéniables qualités, et sous l’influence du témoignage qui lui avait été rendu la veille et qui montrait son caractere sous un jour si favorable, ce respect se transformait en quelque chose d’une nature plus amicale.
 PREDICTED: Le respect faisait par la conviction de ses , quoique de première , n ’ avait pas de temps de temps si gai , et de temps de l ’ avoir été si gai .
--------------------------------------------------------------------------------
    SOURCE: By way of decoration for the apartment, hanging to a nail in the middle of the wall, whose green paint scaled off from the effects of the saltpetre, was a crayon head of Minerva in gold frame, underneath which was written in Gothic letters "To dear Papa."
    TARGET: Il y avait, pour décorer l’appartement, accrochée à un clou, au milieu du mur dont la peinture verte s’écaillait sous le salpêtre, une tête de Minerve au crayon noir, encadrée de dorure, et qui portait au bas, écrit en lettres gothiques: «À mon cher papa.»
 PREDICTED: À propos de cette chambre , à un clou , au milieu de la muraille , dont de la petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite petite , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , , ,
--------------------------------------------------------------------------------
Processing Epoch 04: 100%|████████████████████████████████████████████████████████████| 1929/1929 [02:07<00:00, 15.17it/s, loss=3.153]
stty: 'standard input': Inappropriate ioctl for device
Average loss of epoch 4 is 3.233156673397778
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
--------------------------------------------------------------------------------
    SOURCE: At the beginning of the second week, the child having told him that the police supposed he had gone over to Belgium, Étienne ventured out of his hole at nightfall.
    TARGET: Au commencement de la seconde semaine, l'enfant lui ayant dit que les gendarmes le croyaient passé en Belgique, Étienne osa sortir de son trou, des la nuit tombée.
 PREDICTED: Au commencement de la seconde semaine , l ' enfant lui dit que la police avait filé a la Belgique , osait la cour , Étienne osa de la nuit .
--------------------------------------------------------------------------------
    SOURCE: "Halloa, Charlie!
    TARGET: -- Hello!
 PREDICTED: -- Hello , Charlie !
--------------------------------------------------------------------------------
Processing Epoch 05: 100%|████████████████████████████████████████████████████████████| 1929/1929 [02:06<00:00, 15.27it/s, loss=3.081]
stty: 'standard input': Inappropriate ioctl for device
Average loss of epoch 5 is 2.984292781186141
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
--------------------------------------------------------------------------------
    SOURCE: Had he lovedher?
    TARGET: L'avait-il aimée?
 PREDICTED: - il ?
--------------------------------------------------------------------------------
    SOURCE: Rosemilly?"
    TARGET: --Mais oui.
 PREDICTED: Mme Rosémilly ?
--------------------------------------------------------------------------------
Processing Epoch 06: 100%|████████████████████████████████████████████████████████████| 1929/1929 [02:05<00:00, 15.38it/s, loss=2.823]
stty: 'standard input': Inappropriate ioctl for device
Average loss of epoch 6 is 2.6599805530579737
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
--------------------------------------------------------------------------------
    SOURCE: Aussi, me levant, j’allai a la fenetre et me mis a regarder dans la rue tres animée a ce moment.
    TARGET: I walked over to the window, and stood looking out into the busy street.
 PREDICTED: Besides , me , I went into the room and I saw the street in the street that was animated .
--------------------------------------------------------------------------------
    SOURCE: "Your name?" said the officer, who covered a part of his face with his cloak.
    TARGET: -- Votre nom? dit l'officier, qui se couvrait une partie du visage avec son manteau.
 PREDICTED: « Votre nom ? dit l ' officier qui couvrit sa figure avec son manteau .
--------------------------------------------------------------------------------
Processing Epoch 07: 100%|████████████████████████████████████████████████████████████| 1929/1929 [02:07<00:00, 15.17it/s, loss=2.499]
stty: 'standard input': Inappropriate ioctl for device
Average loss of epoch 7 is 2.388941284533669
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
--------------------------------------------------------------------------------
    SOURCE: Milady had told the truth--her head was confused, for her ill-arranged plans clashed one another like chaos.
    TARGET: Milady avait dit la vérité, elle avait la tête lourde; car ses projets mal classés s'y heurtaient comme dans un chaos.
 PREDICTED: Milady avait dit la vérité , la tête s ' embarrassait , car elle mal aux projets de un tel chaos .
--------------------------------------------------------------------------------
    SOURCE: Soon it became a flight; every house hooted him as he passed, they hastened on his heels, it was a whole nation cursing him with a voice that was becoming like thunder in its overwhelming hatred.
    TARGET: Bientôt, ce fut une fuite, chaque maison le huait au passage, on s'acharnait sur ses talons, tout un peuple le maudissait d'une voix peu a peu tonnante, dans le débordement de la haine.
 PREDICTED: Bientôt elle fut une fuite ; chaque maison le prit , on se hâta de les talons , tout le renversa sa gueule en faisant une sorte de sa voix qui se désespérait .
--------------------------------------------------------------------------------
Processing Epoch 08: 100%|████████████████████████████████████████████████████████████| 1929/1929 [02:04<00:00, 15.46it/s, loss=2.195]
stty: 'standard input': Inappropriate ioctl for device
Average loss of epoch 8 is 2.172951136987527
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
--------------------------------------------------------------------------------
    SOURCE: The others had rugged black surfaces, measured up to fifteen centimeters in width, and were ten or more years old.
    TARGET: Les autres, à surface rude et noire, vieilles de dix ans et plus, mesuraient jusqu'à quinze centimètres de largeur.
 PREDICTED: Les autres étaient , de quinze centimètres , et douze ans plus âgée .
--------------------------------------------------------------------------------
    SOURCE: The account of his connection with the Pemberley family was exactly what he had related himself; and the kindness of the late Mr. Darcy, though she had not before known its extent, agreed equally well with his own words.
    TARGET: Ce qui concernait les rapports de Wickham avec la famille de Pemberley et la bienveillance de Mr. Darcy pere a son égard correspondait exactement a ce que Wickham en avait dit lui-meme.
 PREDICTED: Le récit de sa famille était exactement ce qu ’ il avait fait toutes ses bontés ; et l ’ âme de Mr . Darcy était déja bien qu ’ elle avait connu .
--------------------------------------------------------------------------------
Processing Epoch 09: 100%|████████████████████████████████████████████████████████████| 1929/1929 [02:06<00:00, 15.30it/s, loss=1.998]
stty: 'standard input': Inappropriate ioctl for device
Average loss of epoch 9 is 2.0071091381239112
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
--------------------------------------------------------------------------------
    SOURCE: "He has wonderful luck, that brother of mine. He had just come into alegacy of twenty thousand francs a year."
    TARGET: --Il a joliment de la chance, mon frère, il vient d'hériter de vingtmille francs de rente.
 PREDICTED: -- Il a une chance heureuse , que mon frère de la mienne , il venait de vingt mille francs par an .
--------------------------------------------------------------------------------
    SOURCE: 'Yes, indeed, madam,' says Robin. 'I have attacked her in form five times since she was sick, and am beaten off; the jade is so stout she won't capitulate nor yield upon any terms, except such as I cannot effectually grant.'
    TARGET: --Oui vraiment, madame, dit Robin, je l'ai attaquée en forme cinq fois depuis qu'elle a été malade, et j'ai été repoussé; la friponne est si ferme qu'elle ne veut ni capituler ni céder à aucuns termes, sinon tels que je ne puis effectivement accorder.
 PREDICTED: -- Oui , madame , dit Robin , je l ' ai attaqué à cinq fois depuis qu ' elle fût , et je suis bien disposé , et si elle n ' a pas le libre .
--------------------------------------------------------------------------------
Processing Epoch 10: 100%|████████████████████████████████████████████████████████████| 1929/1929 [02:06<00:00, 15.19it/s, loss=1.824]
stty: 'standard input': Inappropriate ioctl for device
Average loss of epoch 10 is 1.9080390397472786
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
--------------------------------------------------------------------------------
    SOURCE: "But why should you wish to persuade me that I feel more than I acknowledge?" "That is a question which I hardly know how to answer.
    TARGET: – Vous etes vraiment cruelle, repartit Elizabeth.
 PREDICTED: -- Mais pourquoi voulez - vous dire que je me suis plus sûr ? c ' est une question que je sache quelle réponse à ma réponse .
--------------------------------------------------------------------------------
    SOURCE: Come, let me see the list of pitiful fellows who have been kept aloof by Lydia's folly."
    TARGET: Allons, faites-moi la liste de ces pitoyables candidats que cette écervelée de Lydia a effarouchés.
 PREDICTED: Allons , allons voir la liste des enfants , qui ont été retenue par la folie .
--------------------------------------------------------------------------------
Processing Epoch 11: 100%|████████████████████████████████████████████████████████████| 1929/1929 [02:06<00:00, 15.19it/s, loss=1.923]
stty: 'standard input': Inappropriate ioctl for device
Average loss of epoch 11 is 1.8769891720599248
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
--------------------------------------------------------------------------------
    SOURCE: I'll give her to you."
    TARGET: Je te la donne.
 PREDICTED: Je vous la remettrai .
--------------------------------------------------------------------------------
    SOURCE: "By the way," resumed de Winter, stopping at the threshold of the door, "you must not, Milady, let this check take away your appetite.
    TARGET: «À propos, reprit de Winter en s'arrêtant sur le seuil de la porte, il ne faut pas, Milady, que cet échec vous ôte l'appétit.
 PREDICTED: -- À propos , reprit Lord de Winter en s ' arrêtant à la porte de la porte , n ' est - ce pas , Milady , votre appétit .
--------------------------------------------------------------------------------
Processing Epoch 12: 100%|████████████████████████████████████████████████████████████| 1929/1929 [02:07<00:00, 15.10it/s, loss=1.918]
stty: 'standard input': Inappropriate ioctl for device
Average loss of epoch 12 is 1.8508596161352409
huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
To disable this warning, you can either:
	- Avoid using `tokenizers` before the fork if possible
	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
--------------------------------------------------------------------------------
    SOURCE: To much confabulation succeeded a sound of scrubbing and setting to rights; and when I passed the room, in going downstairs to dinner, I saw through the open door that all was again restored to complete order; only the bed was stripped of its hangings. Leah stood up in the window-seat, rubbing the panes of glass dimmed with smoke.
    TARGET: Après ces exclamations, on remit tout en état. Lorsque je descendis pour dîner, la porte de la chambre était ouverte et je vis que le dégât avait été réparé; le lit seul restait encore dépouillé de ses rideaux; Leah était occupée à laver le bord des fenêtres noirci par la fumée; je m'avançai pour lui parler, car je désirais connaître l'explication donnée par M. Rochester; mais en approchant j'aperçus une seconde personne: elle était assise près du lit, et occupée à coudre des anneaux à des rideaux.
 PREDICTED: tout d ' un bruit de et de en se suivant , quand je la chambre des noces , en venant par toute la portière et je ne vis que ses fenêtres .
--------------------------------------------------------------------------------
    SOURCE: Dans cette salle, et tout au fond, un seul étudiant, penché sur une table, completement absorbé par son travail.
    TARGET: At the sound of our steps he glanced round and sprang to his feet with a cry of pleasure.
 PREDICTED: In this room , and quite as to fond , a single peine upon one table , off his so .
--------------------------------------------------------------------------------
Processing Epoch 13: 100%|████████████████████████████████████████████████████████████| 1929/1929 [02:06<00:00, 15.29it/s, loss=1.789]
```


