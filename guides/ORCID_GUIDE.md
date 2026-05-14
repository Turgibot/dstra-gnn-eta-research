# ORCID ID Guide

## What is an ORCID ID?

**ORCID** stands for **Open Researcher and Contributor ID**. It's a unique, persistent identifier (like a digital name tag) for researchers and academics.

### Format
An ORCID ID is a 16-digit number formatted as: `0000-0000-0000-0000`

Example: `0000-0001-2345-6789`

### Why is it Important?

1. **Unique Identification**: Distinguishes you from other researchers with similar or identical names
2. **Persistent Record**: Stays with you throughout your career, even if you change institutions or names
3. **Research Attribution**: Ensures your publications, grants, and contributions are correctly linked to you
4. **Publisher Requirements**: Many journals (including Springer Nature) now require or strongly recommend ORCID IDs
5. **Reduced Data Entry**: Enter your information once, reuse it across platforms and submissions

## Why SN Computer Science Requires It

According to the [SN Computer Science submission guidelines](https://link.springer.com/journal/42979/submission-guidelines):

> **Author information**
> - If available, the 16-digit ORCID of the author(s)

While it's not strictly mandatory (they say "if available"), it's **strongly recommended** because:
- It helps with proper attribution
- It's becoming a standard in academic publishing
- It makes the submission process smoother
- It helps track your research contributions

## How to Get an ORCID ID

### Step 1: Register (Takes ~2 minutes)
1. Go to **https://orcid.org**
2. Click **"Register for an ORCID iD"**
3. Fill in:
   - Your name (as it appears on publications)
   - Email address
   - Password
4. Click **"Register"**

### Step 2: Verify Your Email
- Check your email inbox
- Click the verification link

### Step 3: Complete Your Profile (Optional but Recommended)
- Add your employment/affiliation
- Add education history
- Add publications (can be done automatically later)

### Step 4: Find Your ORCID ID
- Your ORCID ID appears at the top of your profile page
- Format: `https://orcid.org/0000-0000-0000-0000`
- The 16-digit number is your ORCID ID

## How to Add ORCID ID to Your Paper

Once you have your ORCID ID, add it to `main.tex`:

### Current Code (with TODO):
```latex
\author*[1]{\fnm{Guy} \sur{Tordjman}}\email{guy.tordjman@openu.ac.il}
% TODO: Add ORCID ID when available

\author[2]{\fnm{Nadav} \sur{Voloch}}\email{nadavv@ruppin.ac.il}
% TODO: Add ORCID ID when available
```

### Updated Code (with ORCID):
```latex
\author*[1]{\fnm{Guy} \sur{Tordjman}}\email{guy.tordjman@openu.ac.il}
\orcid{0000-0000-0000-0000}  % Replace with your actual ORCID ID

\author[2]{\fnm{Nadav} \sur{Voloch}}\email{nadavv@ruppin.ac.il}
\orcid{0000-0000-0000-0000}  % Replace with actual ORCID ID
```

## Important Notes

1. **Both Authors Need ORCID IDs**: If you're submitting as co-authors, ideally both should have ORCID IDs
2. **Free to Register**: ORCID registration is completely free
3. **Takes Minutes**: The registration process is very quick
4. **Not Mandatory but Recommended**: While SN Computer Science says "if available," having one is best practice
5. **Privacy**: You control what information is public on your ORCID profile

## Benefits for Your Submission

- ✅ **Professional**: Shows you follow current academic publishing standards
- ✅ **Attribution**: Ensures your work is correctly attributed to you
- ✅ **Future Submissions**: Can reuse your ORCID ID for future papers
- ✅ **Grant Applications**: Many funding agencies also use ORCID
- ✅ **Research Profile**: Helps build your online research presence

## Quick Links

- **Register**: https://orcid.org/register
- **Sign In**: https://orcid.org/signin
- **ORCID Support**: https://support.orcid.org
- **What is ORCID?**: https://orcid.org/about/what-is-orcid

## Summary

**ORCID ID** = Your unique academic identifier (like a social security number for researchers)

- **Format**: 16 digits: `0000-0000-0000-0000`
- **Cost**: Free
- **Time to Get**: ~2 minutes
- **Required?**: Recommended (not strictly mandatory for SN Computer Science, but best practice)
- **Where to Get**: https://orcid.org

Once you have your ORCID ID, simply replace the TODO comments in `main.tex` with your actual ORCID numbers using the `\orcid{...}` command.

