/* Principal Matching */
WITH best_fuzzy AS (
    SELECT 
        p."Full School Name" as lead_name
        ,p."Location State" as lead_state
        ,n.nces_id
        ,n.name
        ,n.district_id
        ,n.district_name
        ,jaro_winkler_similarity(
                LOWER(TRIM(p."Full School Name")), LOWER(TRIM(n.name))
            ) as similarity_rating
        ,ROW_NUMBER() OVER (
            PARTITION BY p."Full School Name", p."Location State"
            ORDER BY jaro_winkler_similarity(
                LOWER(TRIM(p."Full School Name")), LOWER(TRIM(n.name))
            ) DESC
        ) as rn
    FROM principal_leads_march_2026 p
    CROSS JOIN nces_public_private_school n
    WHERE n.state_abbr = p."Location State"
      AND jaro_winkler_similarity(
            LOWER(TRIM(p."Full School Name")), LOWER(TRIM(n.name))
          ) > 0.8545
)
SELECT "First Name" as "First Name"
    ,"Last Name" as "Last Name"
    ,coalesce(n1.district_name, n2.district_name, bf.district_name) as "Account Name"
    ,"Email Address" as "Email"
    ,"Title" as "Title"
    ,p."Phone" as "Phone"
    ,"Location Address" as "Mailing Street"
    ,"Location City" as "Mailing City"
    ,"Location State" as "Mailing State"
    ,"Location Zip" as "Mailing Zip"
    ,'US' as "Mailing Country"
    ,'Purchase List - March 26' as "Tag"
    ,'Sales Pre-Qualified' as "Lead Status"
    ,coalesce(n1.district_name, n2.district_name, bf.district_name) as "District"
    ,'Lead' as "Contact Type"
    ,coalesce(n1.district_id, n2.district_id, bf.district_id) as "NCES District ID"
    ,coalesce(n1.nces_id, n2.nces_id, bf.nces_id) as "NCES School ID"
FROM principal_leads_march_2026 p
LEFT JOIN nces_public_private_school n1
    ON LOWER(TRIM(p."Full School Name")) = LOWER(TRIM(n1.name))
    AND p."Location State" = n1.state_abbr
    AND p."Location Zip" = n1.zip
LEFT JOIN (
    SELECT DISTINCT ON (name, state_abbr) *
    FROM nces_public_private_school
    ORDER BY name, state_abbr, nces_id
) n2 ON n1.nces_id IS NULL
     AND LOWER(TRIM(p."Full School Name")) = LOWER(TRIM(n2.name))
     AND p."Location State" = n2.state_abbr
LEFT JOIN (SELECT * FROM best_fuzzy WHERE rn = 1) bf
    ON n1.nces_id IS NULL AND n2.nces_id IS NULL
    AND p."Full School Name" = bf.lead_name
    AND p."Location State" = bf.lead_state

order by similarity_rating
;

/* Superintendent Matching */

WITH best_fuzzy AS (
    SELECT 
        s."District Name" as lead_name
        ,s."Location State" as lead_state
        ,d.district_id
        ,d.name
        ,jaro_winkler_similarity(
                LOWER(TRIM(s."District Name")), LOWER(TRIM(d.name))
            ) as similarity_rating
        ,ROW_NUMBER() OVER (
            PARTITION BY s."District Name", s."Location State"
            ORDER BY jaro_winkler_similarity(
                LOWER(TRIM(s."District Name")), LOWER(TRIM(d.name))
            ) DESC
        ) as rn
    FROM superintendent_leads_march_2026 s
    CROSS JOIN districts_dioceses d
    WHERE d.state_abbr = s."Location State"
      AND jaro_winkler_similarity(
            LOWER(TRIM(s."District Name")), LOWER(TRIM(d.name))
          ) > 0.80
)
SELECT "First Name" as "First Name"
    ,"Last Name" as "Last Name"
    ,coalesce(d1.name, d2.name, bf.name) as "Account Name"
    ,"Email Address" as "Email"
    ,"Title" as "Title"
    ,s."Phone" as "Phone"
    ,"Location Address" as "Mailing Street"
    ,"Location City" as "Mailing City"
    ,"Location State" as "Mailing State"
    ,"Location Zip" as "Mailing Zip"
    ,'US' as "Mailing Country"
    ,'Purchase List - March 26' as "Tag"
    ,'Sales Pre-Qualified' as "Lead Status"
    ,coalesce(d1.name, d2.name, bf.name) as "District"
    ,'Lead' as "Contact Type"
    ,coalesce(d1.district_id, d2.district_id, bf.district_id) as "NCES District ID"
FROM superintendent_leads_march_2026 s
LEFT JOIN districts_dioceses d1
    ON LOWER(TRIM(s."District Name")) = LOWER(TRIM(d1.name))
    AND s."Location State" = d1.state_abbr
    AND substr(s."Location Zip"::string,1,5) = substr(d1.zip::string,1,5)
LEFT JOIN (
    SELECT DISTINCT ON (name, state_abbr) *
    FROM districts_dioceses
    ORDER BY name, state_abbr, district_id
) d2 ON d1.district_id IS NULL
     AND LOWER(TRIM(s."District Name")) = LOWER(TRIM(d2.name))
     AND s."Location State" = d2.state_abbr
LEFT JOIN (SELECT * FROM best_fuzzy WHERE rn = 1) bf
    ON d1.district_id IS NULL AND d2.district_id IS NULL
    AND s."District Name" = bf.lead_name
    AND s."Location State" = bf.lead_state

;

/* DISTRICT LEADS */

SELECT
    "First Name" as "First Name"
    ,"Last Name" as "Last Name"
    ,dd.name as "Account Name"
    ,"Email Address" as "Email"
    ,"Title" as "Title"
    ,dl."Phone" as "Phone"
    ,"Location Address" as "Mailing Street"
    ,"Location City" as "Mailing City"
    ,"Location State" as "Mailing State"
    ,"Location Zip" as "Mailing Zip"
    ,'US' as "Mailing Country"
    ,'Purchase List - March 26' as "Tag"
    ,'Sales Pre-Qualified' as "Lead Status"
    ,dd.name as "District"
    ,'Lead' as "Contact Type"
    ,"District Id" as "NCES District ID"
FROM district_leads_march_2026 dl
left join districts_dioceses dd
    on dl."District Id" = dd.district_id

;

select * from district_leads_march_2026