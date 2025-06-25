import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load model and label encoder
@st.cache_resource
def load_model():
    model = joblib.load("rf.pkl")  # or "svc_model.pkl" if you're using SVC
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder


model, le = load_model()

precaution_dict = {
    "Accessory Breast Tissue (Polymastia)": [
    "Regular self-breast examinations to monitor for lumps or changes",
    "Avoid tight-fitting bras or clothing to reduce irritation",
    "Seek medical evaluation for any pain or changes in breast tissue",
    "Consider surgical consultation if tissue causes discomfort or cosmetic concern"
  ],
  "Adenomyosis": [
    "Maintain a healthy weight to reduce estrogen levels",
    "Use heat therapy (e.g., heating pads) for pain relief",
    "Avoid high-estrogen foods and manage stress",
    "Follow prescribed hormone therapies or medications"
  ],
  "Amastia": [
    "Genetic counseling if planning pregnancy (if hereditary cause suspected)",
    "Use prosthetics or reconstructive options as needed for body image and symmetry",
    "Monitor for any associated chest wall or rib abnormalities",
    "Regular follow-up with a healthcare provider for developmental assessment"
  ],
  "Amniotic Fluid Embolism": [
    "Ensure delivery occurs in a well-equipped medical facility",
    "Discuss any personal or family history of clotting disorders with your provider",
    "Immediate reporting of symptoms like shortness of breath or seizures during labor",
    "Avoid unnecessary trauma or interventions during labor when possible"
  ],
    "Anovulation": [
        "Maintain a balanced diet and healthy body weight",
        "Manage stress levels with relaxation techniques and regular sleep",
        "Track menstrual cycle regularly to detect irregularities",
        "Follow prescribed treatments like ovulation induction medications or hormone therapy"
    ],
    "Asherman’s Syndrome": [
        "Avoid unnecessary uterine surgeries (e.g., multiple D&Cs)",
        "Prompt treatment of uterine infections to prevent scarring",
        "Follow post-surgical care carefully after any uterine procedure",
        "Regular monitoring of menstrual cycle and report abnormalities to your doctor"
    ],
    "Autoimmune Oophoritis": [
        "Regularly monitor hormone levels and ovarian function",
        "Seek early evaluation for menstrual irregularities or infertility",
        "Consider hormone replacement therapy if premature ovarian failure occurs",
        "Work with an endocrinologist to manage associated autoimmune conditions (e.g., Addison’s disease)"
    ],
    "Bacterial Vaginosis": [
        "Avoid douching and scented vaginal products",
        "Practice safe sex and limit number of sexual partners",
        "Maintain proper genital hygiene",
        "Seek medical treatment promptly if symptoms arise"
    ],
    "Bartholin Gland Hyperplasia": [
        "Monitor for persistent gland swelling",
        "Seek medical evaluation for unusual growths or discomfort",
        "Maintain good vulvar hygiene",
        "Avoid trauma or pressure to the vulvar area"
    ],
    "Bartholin’s Cyst": [
        "Practice good vulvar hygiene",
        "Use warm compresses to promote drainage",
        "Avoid tight-fitting underwear and clothing",
        "Seek prompt treatment if pain, swelling, or infection develops"
    ],
    "Benign Breast Disease": [
        "Perform regular breast self-exams",
        "Limit caffeine and dietary fat if recommended",
        "Attend regular clinical breast exams and imaging",
        "Report any new or unusual breast changes"
    ],
    "Benign Proliferative Breast Disease": [
        "Schedule regular mammograms and follow-ups",
        "Limit alcohol intake and maintain a healthy weight",
        "Discuss family history of breast disease with your doctor",
        "Consider lifestyle modifications to reduce risk"
    ],
    "BIA-ALCL (Breast Implant-Associated Anaplastic Large Cell Lymphoma)": [
        "Schedule regular checkups post-implantation",
        "Monitor for breast swelling, lumps, or fluid accumulation",
        "Choose textured implants cautiously and with full risk disclosure",
        "Report any changes around implants promptly to a specialist"
    ],
    "Breast Abscess": [
        "Continue breastfeeding or pumping to prevent milk stasis",
        "Practice good breast hygiene",
        "Seek early treatment for mastitis to prevent abscess formation",
        "Complete full course of antibiotics if prescribed"
    ],
    "Breast Cancer": [
        "Attend routine screenings such as mammograms",
        "Maintain a healthy weight and active lifestyle",
        "Limit alcohol and avoid tobacco",
        "Know your family history and consider genetic counseling if indicated"
    ],
    "Breast Cysts": [
        "Perform regular self-breast exams to detect changes",
        "Reduce caffeine intake if symptomatic",
        "Follow up on abnormal imaging or persistent lumps",
        "Use supportive bras to reduce discomfort"
    ],
    "Breast Engorgement": [
        "Breastfeed or pump frequently to relieve pressure",
        "Apply warm compresses before and cold compresses after feeds",
        "Wear a supportive bra",
        "Massage breasts gently during feeding or pumping"
    ],
    "Breast Fat Necrosis": [
        "Avoid trauma to the breasts when possible",
        "Monitor any lumps or changes post-surgery or injury",
        "Undergo imaging to distinguish from malignancy if needed",
        "Follow up regularly with a healthcare provider"
    ],
    "Breast Hematoma": [
        "Apply cold compresses to reduce swelling initially",
        "Avoid heavy lifting or trauma post-surgery",
        "Monitor for increasing pain or bruising",
        "Follow up with imaging if symptoms persist"
    ],
    "Breast Myofibroblastoma": [
        "Have any breast lump evaluated with imaging and/or biopsy",
        "Monitor for recurrence if removed",
        "Schedule follow-ups with your healthcare provider",
        "Report any new breast symptoms promptly"
    ],
    "Brenner Tumor (Ovarian)": [
        "Attend routine pelvic exams",
        "Seek evaluation for pelvic mass or pain",
        "Follow up with imaging if ovarian masses are detected",
        "Consider surgical removal if symptomatic or enlarging"
    ],
    "Candidiasis (Vaginal Yeast Infection)": [
        "Maintain proper genital hygiene and keep the area dry",
        "Avoid use of scented soaps or douches",
        "Wear breathable, cotton underwear and avoid tight clothing",
        "Control blood sugar levels if diabetic"
    ],
    "Cervical Cancer": [
        "Get regular Pap smears and HPV tests",
        "Get vaccinated against HPV",
        "Avoid smoking and practice safe sex",
        "Follow up on abnormal test results promptly"
    ],
    "Cervical Ectropion": [
        "Avoid irritants like douches or perfumed products",
        "Use protection during intercourse to reduce infection risk",
        "Attend regular pelvic exams",
        "Consult a gynecologist if you notice unusual discharge or bleeding"
    ],
    "Cervical Incompetence": [
        "Discuss prior pregnancy losses with a healthcare provider",
        "Get regular cervical length monitoring during pregnancy",
        "Avoid strenuous activity or heavy lifting during pregnancy",
        "Consider cerclage (a cervical stitch) if recommended"
    ],
    "Cervical Intraepithelial Neoplasia (CIN)": [
        "Get regular cervical screenings (Pap and HPV tests)",
        "Avoid smoking to reduce progression risk",
        "Comply with follow-up appointments and treatments",
        "Practice safe sex to reduce HPV reinfection"
    ],
    "Cervicitis": [
        "Practice safe sex using condoms",
        "Complete all prescribed treatments for STIs",
        "Avoid vaginal irritants and douching",
        "Ensure sexual partners are also treated if needed"
    ],
    "Chronic Endometritis": [
        "Treat infections promptly to prevent complications",
        "Use barrier methods of contraception to reduce infection risk",
        "Get screened if experiencing abnormal bleeding or discharge",
        "Follow complete antibiotic courses as prescribed"
    ],
    "Clitoral Hypertrophy": [
        "Avoid exposure to androgenic substances (medications, supplements)",
        "Have endocrine evaluations if changes are noted",
        "Consult a specialist for congenital causes or anatomical concerns",
        "Avoid trauma or unnecessary manipulation of the area"
    ],
    "Clitoromegaly": [
        "Monitor hormonal levels regularly if diagnosed with hormonal imbalances",
        "Avoid exogenous androgen or steroid use",
        "Seek genetic or endocrinologic evaluation for underlying causes",
        "Consider surgical options only when medically advised"
    ],
    "Corpus Luteum Cyst": [
        "Get regular pelvic ultrasounds if experiencing pain or irregular cycles",
        "Avoid high-impact activities if cysts are large (to prevent rupture)",
        "Monitor symptoms and report severe pain or bleeding",
        "Follow up with gynecologist as recommended"
    ],
    "Decidualized Endometriosis": [
        "Attend all prenatal or post-surgical appointments",
        "Monitor for signs of cyst rupture or torsion during pregnancy",
        "Report sudden pelvic pain to your provider",
        "Limit strenuous activities if advised by a doctor"
    ],
    "DES Daughter Syndrome": [
        "Have regular gynecological exams including Pap and pelvic exams",
        "Inform healthcare provider of DES exposure history",
        "Report any menstrual irregularities or fertility issues",
        "Consider fertility counseling if attempting pregnancy"
    ],
    "Ductal Carcinoma In Situ (DCIS)": [
        "Attend all follow-up mammograms and screenings",
        "Limit alcohol and avoid tobacco use",
        "Follow treatment plans including surgery or radiation as advised",
        "Maintain a healthy weight and active lifestyle"
    ],
    "Dysmenorrhea (pathological)": [
        "Identify and treat the underlying condition (e.g., endometriosis)",
        "Use NSAIDs under guidance for pain management",
        "Track symptoms and menstrual patterns",
        "Consult a gynecologist if symptoms worsen or affect daily life"
    ],
    "Eclampsia": [
        "Attend all prenatal visits for blood pressure and urine checks",
        "Report symptoms like headaches, vision changes, or swelling immediately",
        "Take prescribed medications to control blood pressure",
        "Rest and avoid stress during pregnancy"
    ],
    "Ectopic Pregnancy": [
        "Seek early prenatal care to confirm implantation site",
        "Avoid smoking and maintain reproductive health",
        "Discuss risks with provider if you’ve had prior ectopic pregnancy",
        "Use contraception effectively if not planning pregnancy"
    ],
    "Endocervical Polyp": [
        "Schedule regular pelvic exams to monitor for recurrence",
        "Report post-coital bleeding or unusual discharge",
        "Avoid irritants or trauma to the cervical area",
        "Follow up after removal for pathology and recurrence"
    ],
    "Endometrial Cancer": [
        "Maintain a healthy weight and manage diabetes",
        "Report postmenopausal bleeding promptly",
        "Get screened if at high risk (family history, HNPCC)",
        "Avoid unopposed estrogen therapy"
    ],
    "Endometrial Hyperplasia": [
        "Follow treatment plans with progestin or other hormone therapies",
        "Maintain a healthy BMI and monitor estrogen levels",
        "Get endometrial sampling if symptoms persist",
        "Monitor menstrual patterns and report abnormalities"
    ],
    "Endometrioma": [
        "Have regular pelvic ultrasounds to monitor cysts",
        "Use hormone therapy to suppress ovarian activity",
        "Avoid high-impact activity if cyst is large",
        "Seek care for severe pelvic pain or fertility concerns"
    ],
    "Endometriosis": [
        "Seek early medical attention if experiencing chronic pelvic pain or painful periods",
        "Maintain a healthy diet and regular exercise routine to manage symptoms",
        "Consider hormonal therapy as recommended by your healthcare provider",
        "Avoid unnecessary surgeries; follow up regularly with a gynecologist"
    ],
    "Estrogen-Secreting Ovarian Tumors": [
        "Undergo regular pelvic exams and imaging if at risk",
        "Report abnormal uterine bleeding or menstrual changes promptly",
        "Follow up with hormonal and tumor marker tests as advised",
        "Discuss fertility preservation and treatment options with a specialist"
    ],
    "Female Androgenic Alopecia": [
        "Avoid harsh hair treatments and heat styling",
        "Maintain a balanced diet rich in iron, zinc, and protein",
        "Consult a dermatologist for early medical therapy (e.g., minoxidil)",
        "Manage underlying hormonal conditions like PCOS"
    ],
    "Female Athlete Triad": [
        "Ensure adequate calorie and nutrient intake for activity level",
        "Monitor menstrual cycle and bone health regularly",
        "Avoid excessive physical training without medical supervision",
        "Seek counseling or support for disordered eating patterns"
    ],
    "Female Genital Herpes": [
        "Practice safe sex and use condoms to reduce transmission",
        "Avoid sexual contact during outbreaks",
        "Take antiviral medication as prescribed",
        "Manage stress and other triggers to reduce recurrence"
    ],
    "Female Genital Prolapse": [
        "Perform regular pelvic floor exercises (Kegels)",
        "Avoid heavy lifting and chronic straining (e.g., from constipation)",
        "Maintain a healthy body weight",
        "Seek early evaluation for pelvic pressure or bulging sensations"
    ],
    "Female Genital Tuberculosis": [
        "Complete the full course of anti-tubercular therapy as prescribed",
        "Ensure regular follow-up with a gynecologist and infectious disease specialist",
        "Practice good hygiene and infection control measures",
        "Get screened if there is a history of pulmonary or systemic TB"
    ],
    "Female Genital Sarcoma": [
        "Report any abnormal vaginal bleeding or pelvic mass promptly",
        "Attend regular follow-ups after treatment to monitor recurrence",
        "Avoid delaying diagnosis of persistent or unusual symptoms",
        "Follow prescribed oncology treatment plans closely"
    ],
    "Female Genital Warts": [
        "Get HPV vaccination to reduce risk of genital warts and related cancers",
        "Use barrier protection during sexual activity",
        "Avoid touching or scratching lesions to prevent spread",
        "Follow up with your healthcare provider for treatment options"
    ],
    "Female Pituitary Adenoma": [
        "Monitor hormone levels regularly if diagnosed",
        "Report vision changes or menstrual irregularities promptly",
        "Adhere to prescribed medications or surgical follow-up",
        "Have regular imaging to assess tumor growth"
    ],
    "Female Pseudohermaphroditism": [
        "Receive multidisciplinary care including endocrinology and genetics",
        "Provide psychosocial support and counseling",
        "Monitor hormone levels and reproductive development",
        "Discuss surgical and gender-affirming options thoroughly with specialists"
    ],
    "Female Urethral Caruncle": [
        "Practice good perineal hygiene",
        "Avoid local trauma or irritation (e.g., harsh soaps)",
        "Use estrogen creams in postmenopausal women if indicated",
        "Seek medical evaluation for bleeding or pain"
    ],
    "Female Urethral Syndrome": [
        "Stay well hydrated and void regularly",
        "Avoid bladder irritants (e.g., caffeine, alcohol, spicy foods)",
        "Practice good genital hygiene",
        "Seek early treatment for urinary symptoms or infections"
    ],
    "Fibroadenoma of the Breast": [
        "Perform regular breast self-exams",
        "Follow up with imaging to monitor any changes",
        "Avoid trauma to the breasts",
        "Report any increase in size or changes to a healthcare provider"
    ],
    "Fibrocystic Breast Changes": [
        "Reduce caffeine and high-fat intake if symptomatic",
        "Use supportive bras to ease discomfort",
        "Apply warm or cold compresses for pain relief",
        "Get regular breast evaluations and imaging if needed"
    ],
    "Gestational Trophoblastic Neoplasia (GTN)": [
        "Monitor hCG levels regularly post-treatment",
        "Use reliable contraception during follow-up",
        "Attend all scheduled oncology visits",
        "Report abnormal bleeding immediately"
    ],
    "Gonadoblastoma": [
        "Have regular pelvic ultrasounds and imaging",
        "Seek genetic counseling if Y chromosome material present",
        "Consider prophylactic gonadectomy if advised",
        "Monitor for virilizing symptoms or hormonal changes"
    ],
    "Granulosa Cell Tumor": [
        "Undergo routine pelvic exams and imaging",
        "Follow up CA-125 or inhibin levels post-treatment",
        "Report bloating or pelvic pain early",
        "Discuss fertility-sparing surgery if needed"
    ],
    "HELLP Syndrome": [
        "Attend all prenatal checkups",
        "Report severe headache, visual changes, or upper abdominal pain",
        "Monitor blood pressure and liver enzymes closely",
        "Seek immediate care if fetal movements decrease"
    ],
    "Hidradenitis Suppurativa (Vulvar)": [
        "Maintain good hygiene in affected area",
        "Avoid tight-fitting clothing and shaving",
        "Use topical or oral antibiotics as prescribed",
        "Consider surgical options for recurrent lesions"
    ],
    "Hyperemesis Gravidarum": [
        "Stay hydrated and monitor weight regularly",
        "Eat small, frequent, bland meals",
        "Use anti-nausea medications as directed",
        "Seek IV hydration or hospitalization if symptoms worsen"
    ],
    "Hyperprolactinemia in Women": [
        "Take prescribed dopamine agonists consistently",
        "Avoid stress and excessive physical exertion",
        "Monitor menstrual cycle and vision regularly",
        "Undergo regular pituitary imaging if needed"
    ],
    "Interstitial Cystitis (female-predominant)": [
        "Avoid trigger foods like caffeine and spicy foods",
        "Use bladder instillations or medications as advised",
        "Practice pelvic floor relaxation techniques",
        "Stay hydrated with water rather than acidic drinks"
    ],
    "Intrahepatic Cholestasis of Pregnancy": [
        "Monitor bile acids and liver function tests regularly",
        "Report itching, especially on palms/soles, promptly",
        "Plan early delivery if advised by physician",
        "Avoid medications that burden the liver"
    ],
    "Lactational Gigantomastia": [
        "Wear well-fitted supportive bras",
        "Apply cold compresses to reduce swelling",
        "Discuss surgical reduction if causing complications",
        "Monitor for infection or mastitis"
    ],
    "Lactational Mastitis": [
        "Continue breastfeeding or pumping to empty breast",
        "Apply warm compresses before feeds",
        "Take full course of antibiotics if prescribed",
        "Ensure proper latch and breastfeeding techniques"
    ],
    "Leiomyosarcoma of the Uterus": [
        "Undergo regular follow-up imaging post-surgery",
        "Report unusual bleeding or pelvic pain",
        "Adhere to oncologist-recommended treatment",
        "Consider hysterectomy if not already performed"
    ],
    "Lichen Sclerosus": [
        "Apply corticosteroid ointments as prescribed",
        "Avoid scented soaps or irritants in the genital area",
        "Wear breathable cotton underwear",
        "Monitor for signs of skin changes or malignancy"
    ],
    "Lobular Carcinoma in Situ (LCIS)": [
        "Schedule regular mammograms and clinical exams",
        "Discuss risk-reducing medications with oncologist",
        "Consider bilateral mastectomy if high risk",
        "Avoid hormone replacement therapy unless essential"
    ],
    "Luteal Phase Defect": [
        "Track ovulation with basal body temperature or LH kits",
        "Use progesterone supplements as directed",
        "Monitor menstrual regularity",
        "Avoid stress, which can impact hormone balance"
    ],
    "Luteinized Unruptured Follicle Syndrome (LUFS)": [
        "Track cycles with ultrasound if undergoing fertility treatment",
        "Avoid self-diagnosis; seek reproductive specialist input",
        "Use ovulation induction meds under supervision",
        "Follow-up after failed ovulation cycles"
    ],
    "Luteoma of Pregnancy": [
        "Monitor via ultrasound if diagnosed",
        "Avoid unnecessary surgical intervention unless symptomatic",
        "Follow up postpartum for spontaneous regression",
        "Report signs of virilization promptly"
    ],
    "Mammary Analog Secretory Carcinoma (Breast)": [
        "Have tumor biopsied and staged by specialists",
        "Undergo surgical excision with clear margins",
        "Attend all follow-up oncology appointments",
        "Discuss role of radiation or systemic therapy"
    ],
    "Mammary Duct Ectasia": [
        "Apply warm compresses for discomfort",
        "Practice good nipple hygiene",
        "Avoid squeezing or expressing discharge",
        "Seek care if mass or redness develops"
    ],
    "Mastitis": [
        "Continue breastfeeding or pumping to relieve engorgement",
        "Take antibiotics if prescribed fully",
        "Use warm compresses before feeding",
        "Wear non-restrictive, supportive bras"
    ],
    "Mayer-Rokitansky-Küster-Hauser (MRKH) Syndrome": [
        "Seek psychological and emotional counseling support",
        "Consider vaginal dilation therapy or surgery if desired",
        "Undergo regular pelvic evaluations",
        "Explore fertility options like surrogacy or IVF with uterus transplant"
    ],
    "Meigs’ Syndrome": [
        "Monitor for abdominal distention and shortness of breath",
        "Undergo imaging for pelvic mass evaluation",
        "Seek surgical treatment to remove ovarian fibroma",
        "Follow up to ensure pleural effusion and ascites resolve"
    ],
    "Menometrorrhagia (if pathological)": [
        "Track bleeding patterns and maintain menstrual diary",
        "Use hormonal treatments as advised by a physician",
        "Monitor hemoglobin levels regularly",
        "Seek evaluation for fibroids or endometrial abnormalities"
    ],
    "Menorrhagia (if pathological)": [
        "Take iron supplements if anemic",
        "Use prescribed medications to reduce bleeding",
        "Avoid aspirin or NSAIDs unless advised",
        "Undergo pelvic ultrasound to assess uterine causes"
    ],
    "Micromastia (if pathological)": [
        "Undergo hormonal evaluation if underdeveloped breasts",
        "Consult a plastic surgeon for augmentation options",
        "Seek psychological counseling for body image support",
        "Ensure adequate nutrition during puberty"
    ],
    "Missed Abortion (Retained Fetal Demise)": [
        "Attend all prenatal scans and checkups",
        "Seek immediate care if no fetal movement",
        "Follow medical or surgical management for evacuation",
        "Allow emotional recovery and consider grief support"
    ],
    "Mittelschmerz (if severe/pathological)": [
        "Track ovulation to anticipate pain",
        "Use NSAIDs for pain relief under guidance",
        "Apply heat therapy during ovulation",
        "Consider hormonal contraceptives to suppress ovulation"
    ],
    "Molar Pregnancy": [
        "Follow serial hCG monitoring post-evacuation",
        "Avoid pregnancy for 6–12 months",
        "Use reliable contraception during follow-up",
        "Seek care for persistent bleeding or elevated hCG"
    ],
    "Mucinous Cystadenocarcinoma (Ovarian)": [
        "Undergo surgical staging and removal of tumor",
        "Follow oncology treatment protocols",
        "Attend routine imaging and blood tests",
        "Discuss fertility preservation if relevant"
    ],
    "Müllerian Anomalies (if causing pathology)": [
        "Get evaluated via ultrasound or MRI",
        "Consider surgical correction if obstructive or symptomatic",
        "Monitor for reproductive complications",
        "Use fertility support services when needed"
    ],
    "Non-Lactational Mastitis": [
        "Maintain proper breast hygiene",
        "Avoid irritants or trauma to breast area",
        "Complete antibiotic therapy if prescribed",
        "Seek evaluation for inflammatory breast cancer if persistent"
    ],
    "Oophoritis": [
    "Take full course of antibiotics as directed.",
    "Avoid sexual activity during active infection.",
    "Report pelvic pain and fever early.",
    "Follow up with ultrasound if symptoms persist."
  ],
  "Ovarian Apoplexy": [
    "Seek emergency care for sudden abdominal pain.",
    "Avoid intense physical activity around ovulation.",
    "Monitor hemoglobin levels post-event.",
    "Consider surgical intervention if internal bleeding occurs."
  ],
  "Ovarian Cancer": [
    "Attend regular pelvic exams and imaging.",
    "Know family history and get genetic counseling if needed.",
    "Report symptoms like bloating and pain early.",
    "Follow oncologist-recommended treatment and surveillance."
  ],
  "Ovarian Cyst Rupture": [
    "Use pain relievers as prescribed for mild symptoms.",
    "Seek emergency care for severe pain or bleeding.",
    "Avoid strenuous exercise during ovulation.",
    "Monitor cysts with regular ultrasounds."
  ],
  "Ovarian Dysgerminoma": [
    "Follow scheduled chemotherapy or surgery as prescribed.",
    "Have regular follow-up scans for recurrence.",
    "Discuss fertility preservation before treatment.",
    "Report new symptoms immediately."
  ],
  "Ovarian Dysgenesis": [
    "Begin hormone replacement therapy as advised.",
    "Monitor bone health with regular DEXA scans.",
    "Seek genetic counseling if part of a syndrome.",
    "Consider reproductive options and psychological support."
  ],
  "Ovarian Fibroma": [
    "Monitor tumor with periodic imaging.",
    "Report signs of Meigs’ syndrome like fluid accumulation.",
    "Undergo surgical removal if large or symptomatic.",
    "Follow up for recurrence after treatment."
  ],
  "Ovarian Germ Cell Tumors": [
    "Complete treatment with surgery and/or chemotherapy.",
    "Get regular imaging and tumor marker testing.",
    "Preserve fertility before treatment if possible.",
    "Report recurrence symptoms such as pelvic pain."
  ],
  "Ovarian Hyperstimulation Syndrome (OHSS)": [
    "Closely follow IVF medication protocol.",
    "Monitor for rapid weight gain and abdominal pain.",
    "Avoid heavy exercise during stimulation cycle.",
    "Seek urgent care for difficulty breathing or severe symptoms."
  ],
  "Ovarian Hyperthecosis": [
    "Monitor hormone levels and symptoms of virilization.",
    "Consider weight management strategies.",
    "Use medications to manage excess androgen symptoms.",
    "Evaluate for coexisting metabolic disorders."
  ],
  "Ovarian Remnant Syndrome": [
    "Report pelvic pain after oophorectomy.",
    "Have imaging or laparoscopic evaluation if symptoms occur.",
    "Avoid repeated surgeries unless necessary.",
    "Follow up with gynecologic oncology if risk of malignancy."
  ],
  "Ovarian Teratoma": [
    "Undergo ultrasound or MRI for diagnosis.",
    "Consider surgical removal if large or symptomatic.",
    "Monitor for torsion or rupture.",
    "Follow histopathological results for malignancy risk."
  ],
  "Ovarian Torsion": [
    "Seek emergency care for sudden pelvic pain.",
    "Avoid heavy physical activity during ovulation.",
    "Monitor known cysts with regular ultrasounds.",
    "Follow up after detorsion surgery to prevent recurrence."
  ],
  "Paget’s Disease of the Nipple": [
    "Monitor for changes in the nipple and areola, such as redness or scaling.",
    "Seek medical evaluation for persistent nipple irritation.",
    "Undergo recommended imaging and biopsy procedures.",
    "Follow treatment plans, which may include surgery."
  ],
  "Pelvic Inflammatory Disease (PID)": [
    "Practice safe sex to reduce risk of sexually transmitted infections.",
    "Seek prompt treatment for any genital infections.",
    "Complete the full course of prescribed antibiotics.",
    "Inform sexual partners to prevent reinfection."
  ],
    "Perinatal Depression (if severe/psychiatric diagnosis)": [
    "Attend regular prenatal and postnatal check-ups.",
    "Communicate openly with healthcare providers about mood changes.",
    "Seek support from mental health professionals.",
    "Engage in support groups or counseling as needed."
  ],
  "Peripartum Cardiomyopathy": [
    "Monitor for symptoms like shortness of breath or swelling during late pregnancy and postpartum.",
    "Limit physical activity as advised by a healthcare provider.",
    "Adhere to prescribed heart medications.",
    "Attend regular cardiac evaluations."
  ],
  "Perineal Tears (Obstetric, if severe/infected)": [
    "Practice perineal massage during late pregnancy to increase elasticity.",
    "Follow proper hygiene practices postpartum.",
    "Monitor for signs of infection, such as increased pain or discharge.",
    "Attend follow-up appointments for wound assessment."
  ],
  "Persistent Genital Arousal Disorder (PGAD)": [
    "Seek evaluation from a healthcare provider for persistent genital sensations.",
    "Avoid known triggers that may exacerbate symptoms.",
    "Engage in stress-reducing activities and therapies.",
    "Consider counseling or therapy for coping strategies."
  ],
  "Phyllodes Tumor (Breast)": [
    "Perform regular breast self-examinations to detect lumps early.",
    "Seek medical evaluation for any new or changing breast masses.",
    "Follow through with recommended imaging and biopsies.",
    "Discuss surgical options with a healthcare provider if diagnosed."
  ],
  "Placenta Accreta": [
    "Attend all scheduled prenatal ultrasounds to monitor placental placement.",
    "Discuss delivery plans with a healthcare provider, as cesarean delivery may be necessary.",
    "Prepare for potential blood transfusions during delivery.",
    "Ensure delivery occurs in a facility equipped for high-risk pregnancies."
  ],
  "Placenta Previa": [
    "Avoid activities that could provoke bleeding, such as heavy lifting or sexual intercourse.",
    "Attend regular prenatal appointments to monitor placental position.",
    "Seek immediate medical attention if bleeding occurs.",
    "Plan for cesarean delivery if placenta previa persists near term."
  ],
  "Placental Abruption": [
    "Avoid smoking and substance use during pregnancy.",
    "Manage chronic conditions like hypertension under medical supervision.",
    "Seek immediate medical care if experiencing abdominal pain or bleeding.",
    "Attend all prenatal appointments for monitoring."
  ],
  "Placental Site Trophoblastic Tumor": [
    "Undergo regular follow-up appointments after molar pregnancy or miscarriage.",
    "Monitor hCG levels as advised by a healthcare provider.",
    "Report any unusual bleeding to a healthcare provider promptly.",
    "Follow treatment plans, which may include surgery or chemotherapy."
  ],
  "Polycystic Ovary Syndrome (PCOS)": [
    "Maintain a balanced diet and regular exercise routine.",
    "Monitor menstrual cycles and report irregularities to a healthcare provider.",
    "Manage blood sugar levels through diet or medication.",
    "Discuss fertility concerns with a specialist if planning pregnancy."
  ],
  "Postmenopausal Bleeding (due to pathology)": [
    "Report any vaginal bleeding after menopause to a healthcare provider.",
    "Undergo recommended diagnostic procedures, such as ultrasound or biopsy.",
    "Follow treatment plans based on the underlying cause.",
    "Attend regular gynecological exams."
  ],
  "Postmenopausal Osteoporosis": [
    "Engage in weight-bearing and muscle-strengthening exercises.",
    "Ensure adequate intake of calcium and vitamin D.",
    "Avoid smoking and excessive alcohol consumption.",
    "Discuss bone density testing with a healthcare provider."
  ],
  "Postpartum Depression (if clinical diagnosis)": [
    "Communicate feelings and mood changes with a healthcare provider.",
    "Seek support from family, friends, or support groups.",
    "Consider counseling or therapy as recommended.",
    "Follow prescribed treatment plans, which may include medication."
  ],
  "Postpartum Hemorrhage (if pathological)": [
    "Attend all prenatal appointments to assess risk factors.",
    "Discuss delivery plans with a healthcare provider, especially if at high risk.",
    "Ensure access to skilled birth attendants during delivery.",
    "Follow postpartum care instructions carefully."
  ],
  "Postpartum Psychosis": [
    "Seek immediate medical attention for severe mood swings or hallucinations after childbirth.",
    "Ensure a safe environment for both mother and baby.",
    "Adhere to prescribed treatment plans, which may include hospitalization.",
    "Engage in ongoing mental health support."
  ],
    "Preeclampsia": [
    "Attend all prenatal appointments to monitor blood pressure and urine protein levels.",
    "Report symptoms like severe headaches or visual changes promptly.",
    "Follow a healthcare provider’s recommendations regarding activity levels and diet.",
    "Take prescribed medications and monitor fetal development closely."
  ],
  "Premature Ovarian Failure": [
    "Discuss hormone replacement therapy options with a healthcare provider.",
    "Monitor bone density regularly.",
    "Seek counseling for emotional support and family planning.",
    "Explore fertility options if desired."
  ],
  "Premenstrual Dysphoric Disorder (PMDD)": [
    "Track menstrual cycles and symptoms to identify patterns.",
    "Implement lifestyle changes, such as regular exercise and stress management.",
    "Consider dietary adjustments to alleviate symptoms.",
    "Discuss medication options with a healthcare provider if necessary."
  ],
  "Primary Ovarian Insufficiency": [
    "Consult with a healthcare provider about hormone replacement therapy.",
    "Monitor bone health through regular screenings.",
    "Seek support for emotional well-being and fertility planning.",
    "Explore assisted reproductive technologies if considering pregnancy."
  ],
  "Puerperal Sepsis": [
    "Maintain proper hygiene during and after childbirth.",
    "Monitor for signs of infection, such as fever or foul-smelling discharge.",
    "Seek prompt medical attention for any concerning symptoms.",
    "Adhere to prescribed antibiotic regimens."
  ],
  "Rectocele (if symptomatic/pathological)": [
    "Engage in pelvic floor strengthening exercises.",
    "Avoid heavy lifting and straining during bowel movements.",
    "Maintain a healthy weight to reduce pressure on pelvic organs.",
    "Discuss surgical options with a healthcare provider if symptoms persist."
  ],
  "Recurrent Pregnancy Loss (if due to pathology)": [
    "Undergo comprehensive medical evaluations to identify underlying causes.",
    "Manage chronic health conditions under medical supervision.",
    "Consider genetic counseling if advised.",
    "Follow treatment plans tailored to specific diagnoses."
  ],
  "Retained Products of Conception": [
    "Seek medical evaluation for persistent bleeding or cramping after miscarriage or childbirth.",
    "Undergo recommended imaging studies to assess uterine contents.",
    "Follow through with medical or surgical management as advised.",
    "Attend follow-up appointments to ensure complete recovery."
  ],
  "Rhabdomyosarcoma of the Vagina (Pediatric)": [
    "Monitor for unusual vaginal discharge or bleeding in children.",
    "Seek prompt medical evaluation for any concerning symptoms.",
    "Follow treatment plans, which may include surgery and chemotherapy.",
    "Engage in regular follow-up care to monitor for recurrence."
  ],
  "Salpingitis": [
    "Practice safe sex to reduce the risk of sexually transmitted infections.",
    "Seek prompt treatment for any pelvic or abdominal pain.",
    "Complete the full course of prescribed antibiotics.",
    "Inform sexual partners to prevent reinfection."
  ],
  "Serous Cystadenoma (Ovarian)": [
    "Undergo regular pelvic examinations and imaging studies.",
    "Report any new or worsening pelvic symptoms to a healthcare provider.",
    "Discuss surgical removal if the cyst is large or symptomatic.",
    "Follow up as advised to monitor for recurrence."
  ],
  "Sheehan’s Syndrome": [
    "Monitor for symptoms like fatigue, low blood pressure, or inability to breastfeed after childbirth.",
    "Seek medical evaluation for hormonal deficiencies.",
    "Adhere to prescribed hormone replacement therapies.",
    "Attend regular follow-up appointments to monitor hormone levels."
  ],
  "Skene’s Gland Cyst/Abscess": [
    "Maintain proper genital hygiene.",
    "Seek medical evaluation for any swelling or discomfort near the urethra.",
    "Follow treatment plans, which may include drainage or antibiotics.",
    "Monitor for recurrence and seek prompt treatment if symptoms return."
  ],
  "Struma Ovarii": [
    "Report symptoms like abdominal pain or signs of hyperthyroidism to a healthcare provider.",
    "Undergo recommended imaging and blood tests.",
    "Discuss surgical removal of the tumor if advised.",
    "Follow up to monitor thyroid function post-treatment."
  ],
  "Thecoma (Ovarian Tumor)": [
    "Attend regular gynecological exams to detect ovarian masses early.",
    "Report any abnormal uterine bleeding to a healthcare provider.",
    "Undergo imaging studies as recommended.",
    "Discuss surgical options if the tumor is symptomatic or growing."
  ],
    "Trichomoniasis": [
    "Practice safe sex to prevent transmission.",
    "Seek prompt treatment if experiencing symptoms like vaginal discharge or irritation.",
    "Complete the full course of prescribed medication.",
    "Inform sexual partners to ensure they are treated simultaneously."
  ],
  "Tubo-Ovarian Abscess": [
    "Seek immediate medical attention for severe pelvic pain or fever.",
    "Adhere to prescribed antibiotic regimens.",
    "Undergo imaging studies to monitor abscess size.",
    "Consider surgical intervention if the abscess does not respond to medication."
  ],
  "Turner Syndrome": [
    "Engage in regular medical evaluations to monitor growth and development.",
    "Discuss hormone replacement therapy with a healthcare provider.",
    "Monitor for associated health conditions, such as heart or kidney issues.",
    "Seek support for fertility planning and psychosocial well-being."
  ],
  "Uterine Adenomyoma": [
    "Seek evaluation for abnormal bleeding or pelvic pain.",
    "Follow recommended imaging (e.g., ultrasound, MRI).",
    "Consider hormonal therapy or surgery if symptomatic.",
    "Attend regular follow-ups for symptom monitoring."
  ],
  "Uterine Atony": [
    "Ensure skilled medical supervision during labor and delivery.",
    "Avoid prolonged labor or overuse of uterotonic agents.",
    "Monitor for postpartum bleeding closely.",
    "Have emergency interventions ready during high-risk deliveries."
  ],
  "Uterine Fibroids": [
    "Attend routine pelvic exams.",
    "Report heavy or prolonged menstrual bleeding to your provider.",
    "Consider lifestyle modifications (e.g., healthy weight).",
    "Explore treatment options like medications or surgery if symptomatic."
  ],
  "Uterine Inversion": [
    "Ensure skilled birth assistance to avoid excessive traction on the umbilical cord.",
    "Respond immediately to heavy postpartum bleeding.",
    "Be aware of risk factors like prior uterine surgery.",
    "Prepare for emergency management in delivery settings."
  ],
  "Uterine Prolapse": [
    "Do pelvic floor exercises regularly (e.g., Kegels).",
    "Avoid heavy lifting or chronic straining.",
    "Maintain a healthy weight.",
    "Seek treatment if experiencing urinary or pelvic pressure symptoms."
  ],
  "Uterine Rupture": [
    "Avoid labor induction in patients with uterine scars unless medically necessary.",
    "Ensure close monitoring during labor for women with prior C-section.",
    "Seek immediate care for sudden abdominal pain or fetal distress.",
    "Plan delivery with a skilled obstetric team if high-risk."
  ],
  "Uterine Septum": [
    "Undergo imaging (e.g., hysterosalpingogram, MRI) if recurrent miscarriage occurs.",
    "Consider surgical correction if septum is linked to infertility or loss.",
    "Monitor pregnancy closely if known anomaly is present.",
    "Consult a reproductive specialist for fertility planning."
  ],
  "Vaginal Adenosis": [
    "Follow up on abnormal Pap smears or vaginal symptoms.",
    "Undergo biopsy if suspicious lesions are detected.",
    "Avoid unnecessary estrogen exposure without medical indication.",
    "Seek gynecologic care if exposed to DES in utero."
  ],
  "Vaginal Agenesis": [
    "Seek evaluation during adolescence if primary amenorrhea occurs.",
    "Consult a specialist in reproductive or reconstructive surgery.",
    "Explore psychosocial counseling for emotional support.",
    "Monitor urinary tract health due to potential anomalies."
  ],
  "Vaginal Atresia": [
    "Identify early in childhood/adolescence with absent menstruation or pelvic pain.",
    "Seek surgical correction from a specialist.",
    "Ensure regular follow-up post-surgery to assess outcomes.",
    "Consider counseling and fertility guidance if needed."
  ],
  "Vaginal Cancer": [
    "Report abnormal vaginal bleeding or discharge.",
    "Avoid smoking and limit HPV exposure.",
    "Get regular Pap and pelvic exams.",
    "Consider HPV vaccination as a preventive strategy."
  ],
  "Vaginal Endometriosis": [
    "Track menstrual symptoms and pain severity.",
    "Use hormonal treatments or surgery as directed.",
    "Practice stress-reducing techniques and pain management.",
    "Attend follow-ups for disease monitoring."
  ],
    "Vaginitis": [
    "Practice proper genital hygiene.",
    "Avoid douching and irritant products.",
    "Wear breathable, cotton underwear.",
    "Seek diagnosis before self-treating vaginal symptoms."
  ],
  "Vulvar Cancer": [
    "Monitor for vulvar lumps, itching, or bleeding.",
    "Avoid HPV exposure and consider vaccination.",
    "Quit smoking to reduce risk.",
    "Attend routine gynecologic exams for early detection."
  ],
  "Vulvar Intraepithelial Neoplasia (VIN)": [
    "Report persistent vulvar itching or lesions.",
    "Follow up abnormal Pap or biopsy results promptly.",
    "Avoid HPV exposure and practice safe sex.",
    "Consider HPV vaccination for prevention."
  ],
  "Vulvar Lichen Planus": [
    "Maintain good vulvar hygiene and avoid irritants.",
    "Use prescribed corticosteroid creams to control flare-ups.",
    "Monitor for signs of scarring or malignancy.",
    "Attend dermatology or gynecology follow-up visits."
  ],
  "Vulvar Lichen Sclerosus": [
    "Use topical steroid therapy as prescribed.",
    "Avoid scratching and use emollients for comfort.",
    "Monitor regularly for skin changes or malignancy.",
    "Wear loose, breathable underwear."
  ],
  "Vulvar Pemphigoid": [
    "Avoid trauma or friction to vulvar area.",
    "Use prescribed immune-modulating treatments.",
    "Monitor for secondary infections or ulcers.",
    "Consult dermatology and gynecology for coordinated care."
  ],
  "Vulvar Vestibulitis": [
    "Avoid tight clothing and irritants.",
    "Use topical anesthetics or anti-inflammatory treatments as advised.",
    "Consider pelvic floor physical therapy.",
    "Seek evaluation for persistent vulvar pain."
  ],
  "Vulvodynia": [
    "Track triggers and avoid known irritants.",
    "Use prescribed topical or oral medications for pain.",
    "Try pelvic floor therapy or behavioral counseling.",
    "Seek care from a vulvar pain specialist if symptoms persist."
  ],
  "Wolffian Duct Remnant Cyst (in females)": [
    "Seek evaluation for pelvic or perineal masses.",
    "Undergo imaging (e.g., ultrasound, MRI) to assess anatomy.",
    "Consider surgical removal if symptomatic.",
    "Monitor for recurrence after treatment."
  ]
}

# Symptoms list (same as training)
symptoms = [
    "heavy_menstrual_bleeding",
    "prolonged_periods",
    "severe_cramping",
    "pelvic_pain",
    "Absence_of_one/both_breasts",
    "Missing_nipple(s)",
    "Congenital_(present_at_birth)",
    "May_accompany_chest_wall_abnormalities",
    "Sudden_shortness_of_breath",
    "Hypotension_(low_blood_pressure)",
    "Coagulopathy_(excessive_bleeding)",
    "Seizures_during_labor",
    "Irregular_or_absent_menstrual_periods"
    "Infertility_or_difficulty_conceiving",
    "Acne_or_excess_facial/body_hair",
    "Lack_of_ovulation_symptoms_(e.g.,_no_mid-cycle_pain)",
    "Light_or_absent_menstrual_periods",
    "Infertility_or_recurrent_miscarriages",
    "Pelvic_pain_or_cramping",
    "Scar_tissue_(intrauterine_adhesions)_seen_on_imaging",
    "Irregular_or_absent_periods",
    "Hot_flashes_or_night_sweats",
    "Infertility",
    "Elevated_FSH_and_autoimmune_markers",
    "Thin_grayish-white_vaginal_discharge",
    "Fishy_vaginal_odor",
    "Vaginal_itching_or_irritation",
    "Burning_during_urination",
    "Labial_swelling",
    "Pain_while_sitting_or_walking",
    "Tenderness_near_the_vaginal_opening",
    "Discomfort_during_intercourse",
    "Painless_lump_near_vaginal_opening",
    "Redness_or_swelling",
    "Discomfort_while_walking",
    "Pain_during_sex_if_infected",
    "Lumpy_or_dense_breast_tissue",
    "Breast_tenderness",
    "Breast_swelling",
    "Pain_that_varies_with_menstrual_cycle",
    "Lump_in_the_breast",
    "Nipple_discharge",
    "Increased_breast_density",
    "Swelling_around_breast_implant",
    "Lump_near_implant",
    "Pain_in_the_breast",
    "Fluid_buildup_around_implant",
    "Painful_lump_in_breast",
    "Redness_and_warmth_in_area",
    "Fever_and_chills",
    "Pus_discharge_from_nipple",
    "Breast_lump_or_mass",
    "Change_in_breast_shape",
    "Skin_dimpling_or_puckering",
    "Smooth,_movable_lump_in_breast",
    "Increase_in_size_before_menstruation",
    "Clear_or_yellow_nipple_discharge",
    "Swollen,_firm_breasts",
    "Tenderness",
    "Shiny,_tight_skin_over_breast",
    "Difficulty_breastfeeding",
    "Firm,_round_lump_in_breast",
    "Bruising_or_skin_redness",
    "Tenderness_in_the_area",
    "Skin_dimpling_or_thickening",
    "Localized_swelling",
    "Purple_or_blue_discoloration",
    "Pain_or_tenderness",
    "Lump_or_firmness_in_breast",
    "Slow-growing,_painless_lump",
    "Well-defined_borders",
    "Usually_asymptomatic",
    "Detected_during_routine_imaging",
    "Abdominal_or_pelvic_pain",
    "Pressure_in_lower_abdomen",
    "Abdominal_mass",
    "Occasional_abnormal_bleeding",
    "Vaginal_itching_and_irritation",
    "Thick,_white,_cottage_cheese-like_vaginal_discharge",
    "Redness_and_swelling_of_the_vulva",
    "Pain_or_burning_during_urination_or_intercourse",
    "Abnormal_vaginal_bleeding_(between_periods,_after_intercourse,_or_after_menopause)",
    "Pelvic_pain",
    "Pain_during_intercourse",
    "Unusual_vaginal_discharge_(watery,_thick,_or_foul-smelling)",
    "Increased_vaginal_discharge",
    "Spotting_after_intercourse",
    "Often_asymptomatic_(no_noticeable_symptoms)",
    "May_cause_postcoital_bleeding",
    "Painless_dilation_of_the_cervix_in_the_second_trimester",
    "Premature_rupture_of_membranes",
    "Preterm_labor_and_delivery",
    "Possible_feeling_of_pelvic_pressure",
    "Usually_asymptomatic_(no_noticeable_symptoms)",
    "Abnormal_cells_detected_during_a_Pap_smear",
    "May_lead_to_cervical_cancer_if_untreated",
    "Diagnosed_through_colposcopy_and_biopsy",
    "Abnormal_vaginal_discharge",
    "mucopurulent",
    "Vaginal_bleeding_between_periods",
    "Painful_urination",
    "Abnormal_uterine_bleeding_(heavy,_prolonged,_or_irregular)",
    "Pelvic_pain_or_discomfort",
    "Recurrent_miscarriages",
    "Visibly_enlarged_clitoris",
    "May_be_congenital_or_acquired",
    "Can_be_associated_with_hormonal_imbalances",
    "Often_asymptomatic,_but_may_cause_discomfort_in_some_cases",
    "Significant_enlargement_of_the_clitoris",
    "Often_associated_with_high_androgen_levels",
    "May_be_a_sign_of_an_underlying_medical_condition",
    "Can_cause_psychological_distress",
    "Pelvic_pain_or_discomfort_(usually_on_one_side)",
    "Delayed_or_missed_menstrual_period",
    "Possible_vaginal_spotting_or_bleeding",
    "If_ruptured,_can_cause_sudden_and_severe_abdominal_pain",
    "Pelvic_pain_that_may_worsen_during_menstruation",
    "Abnormal_uterine_bleeding",
    "Formation_of_endometriomas_(chocolate_cysts)_with_decidual_changes_during_pregnancy",
    "Vaginal_adenosis_(glandular_tissue_in_the_vagina)",
    "Cervical_abnormalities_(T-shaped_uterus,_cervical_hood)",
    "Increased_risk_of_clear_cell_adenocarcinoma_of_the_vagina_and_cervix",
    "Infertility_or_difficulty_carrying_a_pregnancy_to_term",
    "May_present_as_a_palpable_breast_lump_or_thickening",
    "Possible_nipple_discharge",
    "Detected_on_mammography_as_calcifications_or_a_mass",
    "Severe_and_debilitating_menstrual_cramps",
    "Pelvic_pain_that_may_radiate_to_the_back_and_thighs",
    "Nausea,_vomiting,_diarrhea",
    "Headache_or_dizziness_during_menstruation",
    "Seizures",
    "High_blood_pressure",
     "Severe_headache",
    "Vision_changes_(blurred_or_double_vision)",
    "Sharp_pelvic_pain",
    "Vaginal_bleeding",
    "Shoulder_pain",
    "Dizziness_or_fainting",
    "Abnormal_vaginal_bleeding",
    "Intermenstrual_bleeding",
    "Postcoital_bleeding",
    "Mucous_vaginal_discharge",
    "Unintended_weight_loss",
    "Heavy_menstrual_bleeding",
    "Irregular_periods",
    "Spotting_between_periods",
    "Postmenopausal_bleeding",
    "Chronic_pelvic_pain",
    "Painful_menstruation",
    "Dysmenorrhea_(painful_periods)",
    "Early_puberty_(in_children)",
    "Enlarged_uterus",
    "Gradual_thinning_of_scalp_hair",
    "Widening_of_the_central_part",
    "Diffuse_hair_loss_on_crown",
    "Preservation_of_frontal_hairline",
    "Amenorrhea_or_irregular_periods",
    "Low_bone_mineral_density_(osteoporosis_risk)",
    "Disordered_eating_or_low_energy_availability",
    "Fatigue_and_decreased_performance",
    "Painful_genital_ulcers",
    "Itching_or_burning_sensation",
    "Dysuria_(painful_urination)",
    "Flu-like_symptoms_(fever,_malaise)",
    "Pelvic_pressure_or_heaviness",
    "Bulge_or_protrusion_from_the_vagina",
    "Urinary_incontinence_or_retention",
    "Difficulty_with_bowel_movements",
    "Irregular_menstrual_cycles",
    "Vaginal_discharge",
    "Pelvic_or_vaginal_mass",
    "Pain_or_discomfort",
    "Urinary_or_bowel_symptoms_(due_to_mass_effect)",
    "Small,_flesh-colored_growths",
    "Itching_or_discomfort",
    "Bleeding_with_intercourse",
    "Clusters_of_cauliflower-like_lesions",
    "Menstrual_irregularities_or_amenorrhea",
    "Galactorrhea_(milk_discharge)",
    "Headaches",
    "Visual_field_defects_(bitemporal_hemianopia)",
    "Ambiguous_genitalia",
    "Enlarged_clitoris",
    "Fusion_of_labia",
    "Normal_female_internal_genitalia",
    "Small,_red,_painful_nodule_at_urethral_opening",
    "Dysuria",
    "Bleeding",
    "Local_irritation_or_itching",
    "Dysuria_without_infection",
    "Frequency_and_urgency",
    "Pelvic_discomfort",
    "Postvoid_dribbling",
    "Firm,_mobile_breast_lump",
    "Non-tender_mass",
    "Well-defined_edges",
    "Usually_solitary_lesion",
    "Lumpy_or_rope-like_breast_texture",
    "Swelling_before_menstruation",
    "Occasional_nipple_discharge",
    "Elevated_hCG_levels",
    "Pelvic_pain_or_pressure",
    "Abdominal_or_pelvic_mass",
    "Signs_of_hormone_secretion",
    "Possible_virilization",
    "Enlarged_ovary",
    "Estrogen_excess_symptoms",
    "Hemolysis_(fatigue,_jaundice)",
    "Elevated_liver_enzymes_(right_upper_quadrant_pain)",
    "Low_platelet_count_(bruising,_bleeding)",
    "Painful_nodules_or_abscesses",
    "Recurrent_flare-ups",
    "Foul-smelling_discharge",
    "Scarring_and_sinus_tract_formation",
    "Severe_nausea_and_vomiting",
    "Dehydration",
    "Weight_loss",
    "Electrolyte_imbalances",
    "Galactorrhea_(milk_production)",
    "Amenorrhea_or_oligomenorrhea",
    "Decreased_libido",
    "Urinary_urgency_and_frequency",
    "Pruritus_(itching),_especially_palms_and_soles",
    "Dark_urine",
    "Light-colored_stools",
    "Jaundice",
    "Rapid,_excessive_breast_enlargement",
    "Breast_pain_and_tenderness",
    "Skin_ulceration_or_thinning",
    "Back_and_shoulder_strain",
    "Breast_pain_and_swelling",
    "Redness_and_warmth_of_the_breast",
    "Fatigue_or_malaise",
    "Pelvic_or_abdominal_pain",
    "Rapidly_enlarging_uterine_mass",
    "Fatigue_or_weight_loss",
    "White,_patchy_skin_around_the_vulva",
    "Painful_intercourse",
    "Skin_tearing_or_bleeding",
    "Non-palpable_abnormal_cells_in_lobules",
    "Found_incidentally_on_biopsy",
    "May_increase_breast_cancer_risk",
    "Short_menstrual_cycles",
    "Spotting_before_menstruation",
    "Infertility_or_early_miscarriage",
    "Low_progesterone_levels",
    "Anovulatory_cycles",
    "Lack_of_temperature_shift_in_basal_body_temperature_charting",
    "Infertility_despite_normal_periods",
    "No_follicular_rupture_on_ultrasound",
    "Ovarian_mass_during_pregnancy",
    "Virilization_in_mother_or_fetus",
    "Regresses_after_delivery",
    "Painless_breast_lump",
    "Slow-growing_mass",
    "Well-circumscribed_tumor_on_imaging",
    "May_involve_axillary_lymph_nodes",
    "Nipple_discharge_(thick,_green_or_black)",
    "Nipple_inversion",
    "Breast_pain_or_tenderness",
    "Palpable_subareolar_mass",
    "Redness_and_warmth_of_breast_skin",
    "Fever_and_malaise",
    "Hard,_painful_lump_in_breast",
    "Primary_amenorrhea",
    "Absent_or_underdeveloped_uterus_and_upper_vagina",
    "Normal_external_genitalia",
    "Normal_development_of_secondary_sexual_characteristics",
    "Ovarian_fibroma",
    "Ascites",
    "Pleural_effusion",
    "Resolution_of_symptoms_after_tumor_removal",
    "Prolonged_bleeding_duration",
    "Fatigue_from_anemia",
    "Excessively_heavy_menstrual_bleeding",
    "Periods_lasting_more_than_7_days",
    "Passing_large_clots",
    "Fatigue_or_shortness_of_breath_(due_to_anemia)",
    "Underdeveloped_breast_tissue",
    "Asymmetry_of_breast_size",
    "Delayed_pubertal_breast_development",
    "Psychological_distress",
    "No_fetal_heartbeat_on_ultrasound",
    "No_symptoms_or_mild_cramping",
    "Brown_vaginal_discharge",
    "Closed_cervical_os",
    "Mid-cycle_pelvic_pain",
    "One-sided_lower_abdominal_discomfort",
    "Pain_lasts_hours_to_a_couple_of_days",
    "May_mimic_appendicitis_or_ectopic_pregnancy",
    "Excessively_high_hCG_levels",
    "Enlarged_uterus_for_gestational_age",
    "Grape-like_vesicles_on_ultrasound",
    "Abdominal_distension",
    "Pelvic_mass",
    "Bloating_and_early_satiety",
    "Constipation_or_urinary_symptoms",
    "Recurrent_pregnancy_loss",
    "Pelvic_pain_or_obstructed_menstruation",
    "Redness_and_warmth_of_the_skin",
    "Purulent_nipple_discharge",
    "Fever_or_chills",
    "Pelvic_or_lower_abdominal_pain",
    "Tenderness_in_the_ovary_region",
    "Irregular_menstruation",
    "Sudden_severe_pelvic_pain",
    "Internal_bleeding_(hemoperitoneum)",
    "Nausea_or_fainting",
    "Abdominal_bloating_or_swelling",
    "Early_satiety_or_appetite_loss",
    "Unexplained_weight_loss",
    "Sharp,_sudden_pelvic_pain",
    "Abdominal_tenderness",
    "Light_vaginal_bleeding",
    "Nausea_or_dizziness",
    "Pelvic_or_abdominal_mass",
    "Pain_or_pressure_in_lower_abdomen",
    "May_cause_precocious_puberty",
    "Underdeveloped_secondary_sexual_characteristics",
    "Streak_ovaries_on_imaging",
    "Pelvic_mass_or_discomfort",
    "Abdominal_bloating",
    "Meigs’_syndrome_(ascites,_pleural_effusion)",
    "Acute_abdominal_pain",
    "Elevated_tumor_markers_(AFP,_hCG)",
    "Menstrual_irregularities",
    "Abdominal_bloating_and_pain",
    "Enlarged_ovaries",
    "Nausea_and_vomiting",
    "Shortness_of_breath_or_fluid_retention",
    "Virilization_(deep_voice,_hirsutism)",
    "Obesity_or_insulin_resistance",
    "Ovarian_enlargement",
    "Pelvic_pain_post-oophorectomy",
    "Mass_in_pelvic_region",
    "Dyspareunia_(pain_during_intercourse)",
    "Urinary_or_bowel_symptoms",
    "Abdominal_pain_or_pressure",
    "May_be_asymptomatic_if_small",
    "Sudden,_severe_lower_abdominal_pain",
    "Adnexal_tenderness",
    "Reduced_or_absent_blood_flow_on_Doppler",
    "Itchy,_scaly_nipple_skin",
    "Nipple_discharge_or_bleeding",
    "Inverted_or_flattened_nipple",
    "May_be_associated_with_underlying_breast_cancer",
    "Lower_abdominal_or_pelvic_pain",
    "Abnormal_vaginal_discharge",
    "Pain_during_intercourse_or_urination",
    "Persistent_sadness_or_hopelessness",
    "Loss_of_interest_in_daily_activities",
    "Fatigue_or_sleep_disturbances",
    "Thoughts_of_self-harm_or_harming_baby",
    "Shortness_of_breath_(especially_lying_down)",
    "Swelling_of_legs_or_ankles",
    "Fatigue_and_weakness",
    "Rapid_heartbeat_or_palpitations",
    "Perineal_pain_or_discomfort",
    "Swelling_or_bruising_of_the_perineum",
    "Foul-smelling_discharge_(if_infected)",
    "Painful_urination_or_defecation",
    "Unwanted_genital_arousal_without_desire",
    "Tingling_or_throbbing_sensations",
    "Prolonged_episodes_lasting_hours_or_days",
    "Emotional_distress_or_anxiety",
    "Rapidly_growing_breast_lump",
    "Firm_and_mobile_mass",
    "Skin_stretching_over_tumor",
    "May_recur_locally_after_removal",
    "Severe_bleeding_during_or_after_delivery",
    "Failure_of_placenta_to_detach_naturally",
    "Preterm_birth_risk",
    "May_require_hysterectomy",
    "Painless_vaginal_bleeding_in_late_pregnancy",
    "Preterm_labor",
    "Abnormal_fetal_position",
    "Detected_on_ultrasound",
    "Sudden,_painful_vaginal_bleeding",
    "Uterine_tenderness_or_contractions",
    "Decreased_fetal_movement",
    "Signs_of_fetal_distress",
    "Irregular_or_heavy_vaginal_bleeding",
    "Pelvic_pain_or_mass",
    "Hirsutism_(excess_body_hair)",
    "Acne_or_oily_skin",
    "Polycystic_ovaries_on_ultrasound",
    "Unexpected_vaginal_bleeding_after_menopause",
    "Spotting_or_light_periods",
    "Associated_with_endometrial_pathology",
    "May_indicate_malignancy",
    "Back_pain_or_spinal_tenderness",
    "Loss_of_height_over_time",
    "Stooped_posture",
    "Increased_risk_of_bone_fractures",
    "Persistent_sadness_or_mood_swings",
    "Withdrawal_from_family_or_baby",
    "Changes_in_appetite_or_sleep",
    "Feelings_of_guilt_or_worthlessness",
    "Heavy_vaginal_bleeding_after_delivery",
    "Drop_in_blood_pressure",
    "Rapid_heart_rate",
    "Hallucinations_or_delusions",
    "Severe_mood_swings",
    "Confusion_or_disorientation",
    "Thoughts_of_harming_self_or_baby",
    "High_blood_pressure_after_20_weeks_gestation",
    "Proteinuria_(protein_in_urine)",
    "Swelling_of_hands_and_face",
    "Headaches_or_visual_disturbances",
    "Missed_or_irregular_periods",
    "Low_estrogen_levels",
    "Severe_mood_swings_before_period",
    "Irritability_or_anger",
    "Fatigue_or_low_energy",
    "Physical_symptoms_like_bloating_or_breast_tenderness",
    "Irregular_or_absent_periods_before_age_40",
    "Hot_flashes",
    "Low_estrogen_and_high_FSH_levels",
    "Fever_within_10_days_of_childbirth",
    "Foul-smelling_vaginal_discharge",
    "Rapid_heart_rate_or_chills",
    "Bulging_sensation_in_the_vagina",
    "Pelvic_pressure_or_fullness",
    "Vaginal_discomfort_during_intercourse",
    "Three_or_more_consecutive_miscarriages",
    "Persistent_vaginal_bleeding_after_delivery_or_miscarriage",
    "Abdominal_cramping",
    "Fever_or_chills_(if_infected)",
    "Enlarged,_tender_uterus",
    "Vaginal_bleeding_or_discharge",
    "Grape-like_mass_protruding_from_vagina",
    "Urinary_or_bowel_obstruction",
    "Lower_abdominal_pain",
    "Purulent_vaginal_discharge",
    "Pressure_symptoms_on_bladder_or_bowel",
    "Usually_asymptomatic_when_small",
    "Failure_to_lactate_postpartum",
    "Low_blood_pressure_or_hypoglycemia",
    "Painful_lump_near_urethral_opening",
    "Difficulty_urinating",
    "Swelling_or_redness_of_the_vulva",
    "Foul-smelling_discharge_(if_abscessed)",
    "Hyperthyroid_symptoms_(palpitations,_weight_loss)",
    "Hot_flashes_or_anxiety",
    "Estrogen_excess_signs",
    "Frothy,_greenish-yellow_vaginal_discharge",
    "Pain_during_urination_or_sex",
    "Strawberry_cervix_on_examination",
    "Severe_pelvic_or_abdominal_pain",
    "Tender_pelvic_mass",
    "Short_stature",
    "Lack_of_secondary_sexual_development",
    "Webbed_neck_or_broad_chest",
    "Heavy_or_prolonged_menstrual_bleeding",
    "Severe_menstrual_cramps",
    "Pelvic_pressure_or_bloating",
    "Heavy_postpartum_bleeding",
    "Soft_or_boggy_uterus_on_exam",
    "Signs_of_shock_(dizziness,_rapid_heartbeat)",
    "Frequent_urination",
    "Constipation_or_back_pain",
    "Severe_vaginal_bleeding_after_childbirth",
    "Visible_uterus_protruding_from_the_vagina",
    "Shock_or_fainting",
    "Bulge_in_or_outside_the_vaginal_opening",
    "Lower_back_pain",
    "Sudden,_severe_abdominal_pain_during_labor",
    "Loss_of_uterine_contractions",
    "Abnormal_uterine_shape_on_imaging",
    "Visible_glandular_epithelium_on_colposcopy",
    "Associated_with_DES_exposure",
    "Absent_or_shortened_vaginal_canal",
    "May_be_associated_with_MRKH_syndrome",
    "Obstructed_menstrual_flow_(hematocolpos)",
    "Lower_abdominal_pain_in_adolescence",
    "Difficulty_with_intercourse",
    "Mass_or_lesion_in_the_vaginal_wall",
    "Cyclic_vaginal_pain_or_bleeding",
    "Tender_vaginal_wall",
    "Associated_pelvic_endometriosis",
    "Abnormal_discharge_(color/odor)",
    "Burning_sensation_during_urination",
    "Redness_or_swelling_of_the_vulva",
    "Persistent_vulvar_itching",
    "Pain_or_burning_in_the_vulvar_area",
    "Lump_or_ulcer_on_vulva",
    "Bleeding_not_related_to_menstruation",
    "Itching_or_burning_in_the_vulva",
    "Visible_lesions_or_white_patches",
    "May_be_asymptomatic",
    "Vulvar_pain_or_burning",
    "White,_lacy_lesions_on_vulva",
     "Erosions_or_ulcers",
    "Pain_during_urination_or_intercourse",
    "Intense_vulvar_itching",
    "Thin,_white,_wrinkled_skin",
    "Painful_cracks_or_sores",
    "Risk_of_scarring_and_narrowing_of_the_vulva",
    "Painful_blisters_on_the_vulva",
    "Skin_erosions_or_ulcers",
    "May_involve_perineal_or_anal_regions",
    "Burning_pain_at_vaginal_opening",
    "Pain_with_tampon_insertion_or_intercourse",
    "Tenderness_on_vestibular_exam",
    "Localized_redness_or_swelling",
    "Chronic_vulvar_pain_without_identifiable_cause",
    "Burning_or_stinging_sensation",
    "Pain_worsened_by_touch_or_pressure",
    "Interference_with_daily_activities_or_intimacy",
    "Urinary_tract_obstruction_(rare)",
    "Often_asymptomatic",
    "Developmental_delay",
    "Hypotonia_(weak_muscle_tone)",
    "Facial_dysmorphisms",
    "Genital_abnormalities_(ambiguous_or_underdeveloped_structures)",
    "Reproductive_tract_anomalies",
    "Urinary_symptoms",
    "Possible_infertility"
    
]

st.title("Female-Specific Disease Predictor")

st.markdown("Select the symptoms you're experiencing:")

# Multi-select symptoms
selected_symptoms = st.multiselect("Symptoms", symptoms)

# Create input vector
input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]

if st.button("Predict Disease"):
    prediction = model.predict([input_vector])[0]
    disease = le.inverse_transform([prediction])[0]
    st.success(f"Predicted Disease: {disease}")
    
    
     # Show precautions
    precautions = precaution_dict.get(disease, ["No specific precautions available for this condition."])
    st.markdown("Recommended Precautions:")
    for i, tip in enumerate(precautions, 1):
        st.markdown(f"{i}. {tip}")