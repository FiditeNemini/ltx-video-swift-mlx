# gemma-4-swift-mlx — demande #5 : fenêtre n-gramme ignorant le canal de pensée

Contexte : votre 1.4.0 (`templateVariables` / `enable_thinking`) est en place côté
ltx-video et fonctionne. Votre prédiction sur l'interaction avec le n-gramme
s'est vérifiée exactement. Matrice à trois branches lancée telle que vous
l'aviez proposée — E2B bf16, même prompt brut, même image, seed 42,
`maxTokens: 1200` :

| Branche | Marqueurs temporels | Formats distincts | Timeline produite |
|---|---|---|---|
| (1) thinking OFF, ngram ON | 6 | 3 | hover `00:500 → 00:14.000` — **déborde le départ annoncé à 07.500** |
| (2) thinking ON, ngram **OFF** | 6 | **1** | hover `00:02.500 → 00:07.000`, départ `00:07.500` — **cohérente** |
| (3) thinking ON, ngram ON | **0** | — | plus **aucun** marqueur |

**(2) répare l'arithmétique temporelle** : bornes conformes au prompt d'entrée,
format homogène.

**(3) l'annule complètement** : le modèle raisonne avec les marqueurs, se les
voit interdits verbatim, et se rabat sur de la prose vague (« at the start of
the sequence », « At the peak of the hover », « Then, exactly as the car
launches »). Zéro marqueur en sortie.

## Demande

Un mode où la fenêtre n-gramme **ignore les tokens du canal de pensée**, en
extension naturelle de `includePromptInWindow` :

```swift
public func chatStreamMultimodal(
    …,
    noRepeatNGramSize: Int? = nil,
    noRepeatNGramIncludesPrompt: Bool = true,
    noRepeatNGramIncludesThinking: Bool = true,   // ← nouveau, défaut = comportement actuel
    templateVariables: [String: any Sendable]? = nil
)
```

Sémantique : quand `false`, les tokens émis entre `<|channel>thought` (ou
`<|think|>`) et `<channel|>` n'alimentent pas l'historique du
`NoRepeatNGramProcessor` — ils sont générés normalement, simplement pas comptés
comme « déjà écrits ». La protection anti-boucle reste active sur le texte de
réponse, qui est ce qu'elle doit protéger.

Détection : côté processor, `didSample` voit passer les ids ; un petit automate
à deux états (dans / hors canal) piloté par les ids des délimiteurs suffit. Ils
sont dans le tokenizer, et vous avez déjà vérifié qu'ils traversent le
détokenizer streaming intacts.

**Cas limite à ne pas rater** : si le canal de pensée n'est jamais fermé
(génération tronquée par `maxTokens`), l'automate doit rester dans l'état « dans
le canal » plutôt que de compter tout le reste. Le comportement dégradé doit
être « pas de blocage », jamais « blocage sur du raisonnement ».

Idéalement le même paramètre sur `chatStream`, pour la symétrie.

## Tests suggérés

- Unitaire processor : avec `includesThinking: false`, une séquence
  `<|channel>thought a b c d e <channel|> a b c d` n'interdit **pas** `e` ;
  avec `true` (défaut), elle l'interdit — comportement actuel.
- Unitaire boucle : hors canal, une répétition interne à la réponse reste
  bloquée dans les deux modes.
- Canal non fermé : la fin de génération à l'intérieur du canal ne fait pas
  planter et n'interdit rien rétroactivement.
- Non-régression : sans thinking, ids et sorties strictement identiques.

## Après merge

Tag. Côté ltx-video : configuration cible **thinking ON + ngram 5 +
`includesThinking: false`**, puis re-mesure sur notre grille habituelle
(marqueurs, homogénéité de format, cohérence hover/départ, adverbe au verbe,
repère de distance) et un run vidéo complet du bench 2CV.

En attendant votre livraison, on tourne en **thinking ON + ngram OFF** (la
branche 2) : la seule configuration qui produit aujourd'hui une timeline
correcte. On accepte le risque de boucle qu'elle réintroduit, mesuré comme nul
sur nos prompts de bench, et documenté comme provisoire.

## Ce qui reste ouvert côté qualité, hors n-gramme

Pour information, deux défauts persistent en (2) et ne relèvent probablement pas
de vous : la caption écrit « medium side profile shot … captured from a
**front-facing** angle » (contradiction interne sur le point de vue) et perd
l'attachement de l'adverbe au verbe de mouvement (« snaps forward … in a violent
burst » au lieu de « snaps **violently** forward »). On les traitera côté prompt
système ou on les documentera comme limite du E2B.
