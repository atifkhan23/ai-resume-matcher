import re
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import io
import base64

# Supported date formats
date_formats = [
    '%Y-%m-%d', '%Y/%m/%d',
    '%Y-%m', '%Y/%m',
    '%d-%m-%Y', '%d/%m/%Y',
    '%b %Y', '%B %Y',
]

def parse_date(date_str: str) -> datetime:
    """Parse different date formats and support 'Present'/'Now'."""
    s = date_str.strip()
    if s.lower() in ('present', 'now'):
        return datetime.now()
    for fmt in date_formats:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unknown date format: {date_str}")


def extract_contact_info(text: str) -> dict:
    """Extract email, phone, GitHub, LinkedIn from text."""
    contact = {}
    email = re.search(r'[\w\.-]+@[\w\.-]+', text)
    if email:
        contact['email'] = email.group()
    phone = re.search(r'(?:\+\d{1,3}[\s-]?)?\d{10,15}', text)
    if phone:
        contact['phone'] = phone.group()
    github = re.search(r'github\.com/\S+', text, re.IGNORECASE)
    if github:
        contact['github'] = github.group()
    linkedin = re.search(r'linkedin\.com/\S+', text, re.IGNORECASE)
    if linkedin:
        contact['linkedin'] = linkedin.group()
    return contact


def extract_experience(text: str) -> list[dict]:
    """Extract experience details including dates, roles, companies."""
    if not text:
        return []
    date_pattern = re.compile(
        r'(?P<start>(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|[A-Za-z]{3,}\.? ?\d{4}|\d{4}[/-]\d{2}))'
        r'\s*(?:-|–|—|to)\s*'
        r'(?P<end>(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|Present|Now|[A-Za-z]{3,}\.? ?\d{4}|\d{4}[/-]\d{2}))',
        re.IGNORECASE
    )
    entries = []
    for line in text.splitlines():
        seg = line.strip()
        if not seg:
            continue
        for m in date_pattern.finditer(seg):
            try:
                start_date = parse_date(m.group('start'))
                end_date = parse_date(m.group('end'))
            except ValueError:
                continue
            duration = (end_date - start_date).days / 365
            label_part = seg[m.end():].strip() or seg[:m.start()].strip()
            label_clean = re.split(r'During|Business|Website|Key|As', label_part)[0].strip(' -–—:')
            label_clean = re.sub(r'^[a-z]+(?:\s+[a-z]+)*,\s*', '', label_clean)
            rc = re.match(r'(?P<role>.+?)\s+at\s+(?P<company>.+)', label_clean, re.IGNORECASE)
            if rc:
                role, company = rc.group('role').strip(), rc.group('company').strip()
            else:
                role, company = label_clean, ''
            entries.append({
                'role': role,
                'company': company,
                'start_date': start_date,
                'end_date': end_date,
                'duration': duration,
            })
    return entries


def extract_by_keywords(text: str, keyword: str) -> str:
    """Extract specific sections like Education, Experience, Skills, Projects."""
    sections = ['Education', 'Experience', 'Skills', 'Projects']
    pat = re.compile(rf'{keyword}', re.IGNORECASE)
    m = pat.search(text)
    if not m:
        return ''
    start = m.end()
    end = len(text)
    for sec in sections:
        if sec.lower() == keyword.lower():
            continue
        m2 = re.search(rf'{sec}', text[start:], re.IGNORECASE)
        if m2:
            end = start + m2.start()
            break
    return text[start:end].strip()


def extract_structured_fields(text: str) -> dict:
    """Extract all structured information from resume text."""
    data = {
        'contact': extract_contact_info(text),
        'education': extract_by_keywords(text, 'Education'),
        'experience': extract_by_keywords(text, 'Experience'),
        'skills': extract_by_keywords(text, 'Skills'),
        'projects': extract_by_keywords(text, 'Projects'),
    }
    data['experience_details'] = extract_experience(data['experience'])
    return data


def generate_experience_image(experience_data) -> str | None:
    """Generate and display a timeline image of experiences with all extracted dates on the X-axis."""
    if isinstance(experience_data, str):
        entries = extract_experience(experience_data)
    else:
        entries = experience_data
    if not entries:
        return None

    jobs = []
    for e in entries:
        s, t = e['start_date'], e['end_date']
        if t < s:
            s, t = t, s
        title = e['role'] + (f" at {e['company']}" if e['company'] else "")
        if len(title.split()) > 6:
            title = ' '.join(title.split()[:6]) + ' ...'
        jobs.append({"title": title, "start": s, "end": t})

    # sort by start
    jobs.sort(key=lambda x: x['start'])

    # numeric conversion
    for job in jobs:
        job['start_num'], job['end_num'] = mdates.date2num(job['start']), mdates.date2num(job['end'])

    fig, ax = plt.subplots(figsize=(14, 6))
    for idx, job in enumerate(jobs):
        ax.plot([job['start_num'], job['end_num']], [idx, idx], marker='o', linewidth=4)

    ax.set_yticks(range(len(jobs)))
    ax.set_yticklabels([job['title'] for job in jobs], fontsize=12)

    # generate monthly ticks between min and max date
    min_date = min(job['start'] for job in jobs)
    max_date = max(job['end'] for job in jobs)
    monthly = pd.date_range(start=min_date, end=max_date, freq='MS')
    tick_nums = mdates.date2num(monthly.to_pydatetime())
    ax.set_xticks(tick_nums)
    ax.set_xticklabels([d.strftime('%b %Y') for d in monthly], rotation=45, ha='right')

    ax.xaxis.set_minor_locator(mdates.DayLocator(interval=15))
    ax.grid(axis='x', which='major', linestyle='--', alpha=0.6)

    ax.set_xlim(min(tick_nums) - 10, max(tick_nums) + 10)
    ax.set_xlabel("Date", fontsize=14)
    ax.set_title("Job Experience Timeline", fontsize=18, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.show()

    total_years = sum((job['end'] - job['start']).days for job in jobs) / 365
    print(f"\nOverall Experience: {total_years:.1f} years\n")
    return base64.b64encode(buf.read()).decode('utf-8')
