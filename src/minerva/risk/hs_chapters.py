"""HS 2-digit chapter titles and descriptions.

Used by the AI risk engine to predict the most likely HS chapter from a
shipment's free-text description via semantic similarity. Comparing the
predicted chapter(s) with the declared HS code surfaces misclassification,
evasion, and coherence issues for the officer's review.

Covers all 99 chapters of the Harmonized System (WCO HS 2022 edition,
2-digit level). Descriptions are intentionally concise but rich in
identifying vocabulary to maximize embedding discriminativeness.
"""

from __future__ import annotations

# Chapter → (short title, embedding description)
HS_CHAPTERS: dict[str, tuple[str, str]] = {
    "01": ("Live animals", "Live animals including cattle, sheep, horses, pigs, poultry, and other livestock"),
    "02": ("Meat and edible meat offal", "Fresh, chilled or frozen meat and edible offal of cattle, pigs, poultry, sheep"),
    "03": ("Fish and seafood", "Fish, crustaceans, molluscs, aquatic invertebrates — fresh, frozen, or processed seafood"),
    "04": ("Dairy, eggs, honey", "Milk, cream, butter, cheese, yoghurt, eggs, natural honey, edible animal products"),
    "05": ("Other animal products", "Human hair, bones, shells, coral, ivory, feathers — products of animal origin not elsewhere classified"),
    "06": ("Live plants, cut flowers", "Live trees, plants, bulbs, roots, cut flowers, ornamental foliage"),
    "07": ("Vegetables", "Edible vegetables, roots, tubers — fresh, chilled, or frozen"),
    "08": ("Fruits and nuts", "Edible fruits and nuts, peel of citrus fruits or melons"),
    "09": ("Coffee, tea, spices", "Coffee, tea, maté, mate, spices including pepper, cinnamon, cloves, nutmeg"),
    "10": ("Cereals", "Wheat, rice, barley, oats, maize, sorghum and other cereal grains"),
    "11": ("Milling products, malt, starch", "Flour, groats, meal, malt, starches, wheat gluten, inulin"),
    "12": ("Oil seeds, industrial plants", "Oil seeds, oleaginous fruits, industrial or medicinal plants, straw, fodder"),
    "13": ("Lac, gums, resins", "Lac, natural gums, resins, balsams, vegetable saps and extracts"),
    "14": ("Vegetable plaiting materials", "Bamboo, rattan, vegetable materials used for plaiting, stuffing, or padding"),
    "15": ("Animal and vegetable fats and oils", "Edible and inedible animal or vegetable fats, oils, waxes — soybean oil, palm oil, tallow"),
    "16": ("Meat and fish preparations", "Preparations of meat, fish, crustaceans — sausages, canned meat, prepared seafood"),
    "17": ("Sugars and confectionery", "Cane sugar, beet sugar, molasses, sugar confectionery, chocolate-free candies"),
    "18": ("Cocoa and cocoa preparations", "Cocoa beans, cocoa paste, cocoa butter, chocolate, chocolate confectionery"),
    "19": ("Cereal, flour, milk preparations", "Pasta, bread, pastry, cakes, biscuits, breakfast cereals, prepared foods of flour"),
    "20": ("Preparations of vegetables, fruit, nuts", "Jams, jellies, marmalades, fruit juices, canned or preserved vegetables and fruit"),
    "21": ("Miscellaneous edible preparations", "Coffee extracts, sauces, soups, broths, ice cream, yeast, food preparations"),
    "22": ("Beverages, spirits, vinegar", "Water, mineral water, beer, wine, spirits, vinegar, alcoholic and non-alcoholic drinks"),
    "23": ("Residues, animal feed", "Residues and waste from food industries, prepared animal feed, oil-cake, bran"),
    "24": ("Tobacco", "Unmanufactured tobacco, cigars, cigarettes, tobacco products, vaping products"),
    "25": ("Salt, sulphur, earths, stone, plaster, cement", "Salt, sulphur, earths and stone, plastering materials, lime, cement"),
    "26": ("Ores, slag, ash", "Metal ores, slag, ash — iron ore, copper ore, aluminium ore, uranium ore, radioactive ores"),
    "27": ("Mineral fuels, oils, waxes", "Petroleum, crude oil, natural gas, coal, coke, petroleum waxes, mineral fuels"),
    "28": ("Inorganic chemicals", "Inorganic chemicals, rare earth compounds, isotopes, radioactive elements, uranium, plutonium"),
    "29": ("Organic chemicals", "Organic chemicals, hydrocarbons, alcohols, ethers, amines, antibiotics precursors"),
    "30": ("Pharmaceutical products", "Medicaments, vaccines, blood products, bandages, dental cements, pharmaceutical products"),
    "31": ("Fertilizers", "Mineral or chemical fertilizers — nitrogenous, phosphatic, potassic"),
    "32": ("Tanning, dyeing extracts, paints, inks", "Tanning extracts, dyes, pigments, paints, varnishes, printing inks"),
    "33": ("Essential oils, perfumery, cosmetics", "Essential oils, perfumes, cosmetics, toiletries, soaps, shampoos"),
    "34": ("Soap, washing preparations, candles", "Soap, detergents, washing preparations, lubricants, waxes, polishes, candles"),
    "35": ("Albuminoidal substances, glues, enzymes", "Casein, albumins, gelatin, peptones, dextrins, starches, glues, enzymes"),
    "36": ("Explosives, pyrotechnics, matches", "Explosives, propellant powders, fuses, detonators, fireworks, matches, pyrotechnic articles"),
    "37": ("Photographic goods", "Photographic or cinematographic film, plates, paper, chemical preparations"),
    "38": ("Miscellaneous chemical products", "Insecticides, herbicides, disinfectants, lubricants, catalysts, miscellaneous chemicals"),
    "39": ("Plastics and articles thereof", "Polymers, plastics in primary forms, plastic sheets, tubes, articles of plastic"),
    "40": ("Rubber and articles thereof", "Natural rubber, synthetic rubber, vulcanised rubber, tyres, rubber articles"),
    "41": ("Raw hides, skins, leather", "Raw hides, skins, tanned or dressed leather, fur skins"),
    "42": ("Articles of leather", "Handbags, suitcases, saddlery, harness, travel goods, articles of leather"),
    "43": ("Furskins and artificial fur", "Raw furskins, tanned furskins, articles of fur, artificial fur"),
    "44": ("Wood and articles of wood", "Fuel wood, lumber, plywood, wood chips, wood articles, charcoal"),
    "45": ("Cork and articles of cork", "Natural cork, agglomerated cork, articles of cork"),
    "46": ("Manufactures of straw, basketware", "Plaited materials, basketware, wickerwork"),
    "47": ("Pulp of wood", "Wood pulp, paper pulp, recovered paper or paperboard"),
    "48": ("Paper and paperboard", "Paper, paperboard, articles of paper — newsprint, kraft paper, cardboard boxes"),
    "49": ("Printed books, newspapers", "Printed books, newspapers, pictures, maps, music scores, printed matter"),
    "50": ("Silk", "Silk, silk yarn, silk fabrics, silkworm cocoons"),
    "51": ("Wool, fine animal hair, horsehair", "Wool, fine or coarse animal hair, horsehair yarn and woven fabric"),
    "52": ("Cotton", "Raw cotton, cotton yarn, cotton fabrics"),
    "53": ("Other vegetable textile fibres", "Flax, jute, hemp, ramie, paper yarn and woven fabrics"),
    "54": ("Man-made filaments", "Synthetic and artificial filament yarn, woven fabrics of man-made filaments"),
    "55": ("Man-made staple fibres", "Synthetic staple fibres, woven fabrics of man-made staple fibres"),
    "56": ("Wadding, felt, nonwovens, twine, ropes", "Wadding, felt, nonwovens, special yarns, twine, cordage, ropes, cables"),
    "57": ("Carpets and other textile floor coverings", "Carpets, rugs, woven floor coverings, knotted or tufted carpets"),
    "58": ("Special woven fabrics, tapestries, lace", "Terry towelling, gauze, tapestries, lace, embroidery, trimmings"),
    "59": ("Impregnated, coated textile fabrics", "Impregnated, coated, covered or laminated textile fabrics, conveyor belts, hose"),
    "60": ("Knitted or crocheted fabrics", "Knitted or crocheted fabrics of textile materials"),
    "61": ("Apparel, knitted or crocheted", "Knitted or crocheted articles of apparel and clothing accessories — T-shirts, sweaters, underwear"),
    "62": ("Apparel, not knitted or crocheted", "Woven articles of apparel — suits, shirts, trousers, dresses, coats"),
    "63": ("Other made-up textile articles", "Bed linen, blankets, curtains, tents, worn clothing, rags, textile articles"),
    "64": ("Footwear, gaiters", "Shoes, boots, sandals, sports footwear, parts of footwear, gaiters"),
    "65": ("Headgear", "Hats, caps, headgear and parts thereof"),
    "66": ("Umbrellas, walking-sticks, riding-crops", "Umbrellas, sun umbrellas, walking sticks, riding-crops, whips"),
    "67": ("Prepared feathers, artificial flowers", "Prepared feathers, articles of feathers, artificial flowers, articles of human hair"),
    "68": ("Articles of stone, plaster, cement", "Articles of stone, plaster, cement, asbestos, mica, building materials"),
    "69": ("Ceramic products", "Bricks, tiles, sanitary ware, porcelain, ceramic tableware, refractory goods"),
    "70": ("Glass and glassware", "Glass in primary forms, glass fibres, glass containers, glassware, optical glass"),
    "71": ("Precious metals, jewellery", "Pearls, precious stones, diamonds, gold, silver, platinum, jewellery, coins"),
    "72": ("Iron and steel", "Pig iron, ingots, flat-rolled products, bars, rods, wires of iron or steel"),
    "73": ("Articles of iron or steel", "Tubes, pipes, nails, screws, springs, structures, containers of iron or steel"),
    "74": ("Copper and articles thereof", "Copper, unwrought and wrought, refined copper, copper articles"),
    "75": ("Nickel and articles thereof", "Nickel, unwrought and wrought, nickel alloys, nickel articles"),
    "76": ("Aluminium and articles thereof", "Aluminium, unwrought and wrought, aluminium alloys, foil, articles of aluminium"),
    "78": ("Lead and articles thereof", "Lead, unwrought and wrought, lead articles"),
    "79": ("Zinc and articles thereof", "Zinc, unwrought and wrought, zinc articles"),
    "80": ("Tin and articles thereof", "Tin, unwrought and wrought, tin articles"),
    "81": ("Other base metals and articles", "Tungsten, molybdenum, tantalum, titanium, zirconium, cobalt, and other base metals"),
    "82": ("Tools, cutlery, hand tools", "Tools, cutlery, spoons, forks, knives, hand tools, saw blades"),
    "83": ("Miscellaneous articles of base metal", "Locks, keys, hinges, safes, fittings, mountings, bells, statuettes of base metal"),
    "84": ("Machinery, nuclear reactors, boilers", "Nuclear reactors, boilers, machinery, mechanical appliances, turbines, engines, pumps, computers"),
    "85": ("Electrical machinery and equipment", "Electrical machinery, electrical equipment, motors, generators, batteries, telephones, integrated circuits, semiconductors, encryption equipment"),
    "86": ("Railway or tramway equipment", "Railway locomotives, rolling stock, tramway vehicles, track fixtures, signalling equipment"),
    "87": ("Vehicles other than railway", "Motor vehicles, tractors, motorcycles, bicycles, trailers, vehicle parts"),
    "88": ("Aircraft, spacecraft", "Aircraft, spacecraft, helicopters, balloons, aircraft parts, satellites, launch vehicles"),
    "89": ("Ships, boats, floating structures", "Cruise ships, cargo ships, warships, yachts, fishing vessels, floating structures"),
    "90": ("Optical, photographic, medical instruments", "Optical instruments, cameras, microscopes, medical and surgical instruments, measuring and checking instruments, lasers, thermal imaging"),
    "91": ("Clocks and watches", "Clocks, watches, clock movements, watch parts, time of day recording apparatus"),
    "92": ("Musical instruments", "Pianos, string instruments, wind instruments, electronic musical instruments"),
    "93": ("Arms and ammunition", "Firearms, weapons, ammunition, military weapons, parts and accessories of weapons"),
    "94": ("Furniture, lighting, prefab buildings", "Furniture, mattresses, lamps, lighting fittings, illuminated signs, prefabricated buildings"),
    "95": ("Toys, games, sports equipment", "Toys, games, sports equipment, fishing rods, playing cards, festival articles"),
    "96": ("Miscellaneous manufactured articles", "Brooms, brushes, pens, pencils, buttons, zippers, vacuum flasks, miscellaneous articles"),
    "97": ("Works of art, collector's pieces, antiques", "Paintings, drawings, sculpture, collections, antiques over 100 years old"),
    "98": ("Special classification provisions", "Special classification provisions (country-specific)"),
    "99": ("Temporary legislation", "Temporary modifications of import duties (country-specific)"),
}


def get_chapter(hs_code: str | None) -> str | None:
    """Extract the 2-digit chapter from an HS code of any length."""
    if hs_code is None:
        return None
    cleaned = hs_code.strip().replace(".", "").replace(" ", "")
    if len(cleaned) < 2 or not cleaned[:2].isdigit():
        return None
    return cleaned[:2]


def chapter_description(chapter: str) -> str | None:
    """Return the human-readable description for a 2-digit chapter."""
    entry = HS_CHAPTERS.get(chapter)
    return entry[1] if entry else None


def chapter_title(chapter: str) -> str | None:
    """Return the short title for a 2-digit chapter."""
    entry = HS_CHAPTERS.get(chapter)
    return entry[0] if entry else None
